# Operations

MODELOME's storage and source jobs are resumable one-shot operations. The current
operating phase is **not** a scheduled global harvest: it is entry-first ingestion of one
paper and its directly supported resources at a time. See
[Entry-first modelome](entry-first.md) for the canonical workflow and non-goals.

`modelome daily` remains an available future corpus-maintenance composition. Its current
feeds, bulk controls, historical bootstraps, and frontier items have independent durable
state; artifact revisions and bulk controls are content-addressed. It should not be made
the default scheduled job until the per-paper entry pipeline has been validated and a
specific source-coverage decision authorizes it.

## Local setup

```bash
uv sync --extra dev
uv run modelome init
uv run modelome sources
uv run modelome benchmarks
uv run modelome stats
uv run modelome status
```

`--max-pages` is a per-source budget and defaults to 1000 for `sync`. One page is
a smoke test, not a complete scan. Global options precede the subcommand:

```bash
uv run modelome --store /var/lib/modelome/store --json status
```

The searchable evidence store defaults to `data/store`; set `MODELOME_STORE` or pass
`--store` to use another directory. Verified bulk shards default to `data/lake`; set
`MODELOME_LAKE` or pass `--lake` independently. Both stores contain Parquet and manifests,
with no SQL service or embedded SQL database. Source configuration defaults to
`config/sources.toml`. Runtime paths and
optional credentials come from `.env.example`. Credentials are sent only in request
headers or API query parameters required by the upstream service; HTTP diagnostic
URLs redact recognized secret parameters, and the provider adapter excludes its
configured token from stored raw evidence and errors.

## Entry-first practice

Use a stable paper identity as the work-unit boundary: DOI, arXiv ID, PMID, PMCID,
OpenAlex work ID, or a source-owned canonical URL. Record what was ingested, its direct
resource links, link locators, and any failures. Do not turn “one paper” into an
unbounded recursive crawl or a download of PDFs, repository clones, datasets, or weight
files.

`ingest-paper --input PAPER.json` is the explicit one-paper composition. Its input is
one normalized source observation with a source-owned identity—not a title search—and
may contain retained text, direct links, source-declared subjects, and loose tags. It
persists the paper under its original source identity, extracts conservative candidates
only when the source supplied none, records every direct text URL with a locator, and
returns a portable entry seed plus a no-side-effect entry plan. It never materializes an
entry corpus, fetches a paper, or recursively crawls the web. Start with a review-only
run that does not open the registry:

```bash
uv run modelome --json ingest-paper --input PAPER.json --dry-run
```

The surrounding bounded operations remain deliberately separate:

- source adapters and `sync` normalize and checkpoint source records;
- `resolve-paper-text` reports lawful text-access routes for one DOI;
- `ingest-text` and `ingest-institutional-text` accept only locally extracted,
  authorized UTF-8 text—never a PDF or browser session;
- `export-entry-seeds --link-current-resources`, `entry-plan`, and
  `build-entry-corpus` define the portable entry seed, exact current-resource join,
  action-plan, and offline exact-ID-first entry-bundle workflow; they remain separate
  from durable source work-item state and do not make source syncs build entries
  implicitly;
- `crawl --limit N --max-depth 1` can enrich an already-discovered public resource link
  under the URL, robots, and size policies; and
- `link-artifacts` remains available when a separately needed, full evidence-backed
  paper/resource relation projection is required.

The command is a composition boundary, not a claim that the user-facing entry corpus is
already published. A corpus-scale runner still needs durable work-item scheduling and
source-specific reconciliation before it becomes the sink for every enumerator. The
required behavior is in [Entry-first modelome](entry-first.md#the-single-paper-ingestion-loop).

For a non-destructive source smoke test only, use a one-page run with bulk loading and
frontier enrichment disabled:

```bash
uv run modelome daily --max-pages 1 --max-new-shards 0 --bootstrap-max-pages 1 --no-frontier
```

That test is not part of routine entry ingestion and does not establish corpus coverage.

## Parquet store lifecycle

The store is an append-only history of immutable commits. A commit directory contains
`manifest.json` and complete logical snapshots under `tables/*.parquet`. A writer stages
and fsyncs the files, renames the commit into `commits/`, and atomically replaces
`HEAD.json` only after the snapshot is complete. Readers load only the commit selected
by `HEAD.json`, so an interruption before publication leaves the prior snapshot visible.
Current v2 manifests carry SHA-256 maps for each logical table and physical Parquet file;
readers verify those before use. Older v1 manifests remain supported with their canonical
whole-state digest validation and migrate on the next successful write.

An advisory `.write.lock` serializes writers across all sources. Schedule only one MODELOME
writer process against a store directory; source run leases provide additional recovery
for interrupted work but do not make concurrent writers useful. Back up the entire
directory, including `HEAD.json` and `commits/`, as one unit. Restoring only individual
Parquet files can produce a snapshot that its manifest does not describe.

If `HEAD.json` is missing while immutable commits remain, MODELOME fails closed instead of
publishing a new empty registry. Restore the complete store from a consistent backup
after inspecting the retained commits. The `commits`, `.staging`, `.write.lock`, HEAD,
manifest, and table paths must be real paths inside the store, not symlinks.

The initial implementation writes a complete logical snapshot on every commit. Monitor
free space and commit-directory growth. A retention-aware compactor is planned; it will
publish a verified self-contained checkpoint under the same lock before removing only
unreferenced physical commits. Until that command exists, do not manually remove the
commit selected by `HEAD.json`. Logical artifact revisions and provenance remain part
of compacted snapshots rather than being discarded with old physical commits.

Inspect immutable bulk state separately:

```bash
uv run modelome lake-status
uv run modelome lake-status --verify-shards
uv run modelome project --source semantic-scholar
```

`lake-status` reports sealed releases, shards, and physical snapshot/diff rows. Those
rows are not a count of deduplicated papers or model entities. The default verifies
release-seal metadata; `--verify-shards` additionally rehashes every referenced Parquet
part and can be expensive. `project` joins the newest complete Semantic Scholar
`papers`, `abstracts`, and `paper-ids` release, then streams that verified corpus through
the generic extractor into a separately sealed model-candidate Parquet projection. For an intentionally multi-lineage lake,
use an exact `--base-artifact-id` or `--target-artifact-id`; the command never guesses.

## Public metadata export

Run the metadata-only export after a coherent source sync and artifact-relation
projection:

    uv run modelome export-metadata --output data/exports/modelome-public-metadata

The command writes a new directory only. It contains model identities, aliases,
external identifiers, artifacts, current model-artifact links, contextual URL links,
releases, and a source manifest. It does not export raw records, abstracts, paper or
README text, model-card bodies, or evidence values. The generated README documents
source-level rights and attribution constraints; review it before uploading to a public
Hub repository. The export receipt records the immutable store commit and hashes every
payload file, except itself to avoid a self-referential hash.

## What the daily run does later

After an explicit decision to operate a complete source pipeline, `modelome daily`
performs eight phases in order: sync every enabled current/control adapter,
transfer bounded bulk shards, project sealed GH Archive hours into exact repository
discoveries, publish any complete Semantic Scholar snapshot/diff projection, resume
complete arXiv and PMC history under `--bootstrap-max-pages`, enrich exact arXiv IDs
through alphaXiv when `ALPHAXIV_API_KEY` is configured, process the URL frontier, and
publish a sealed current artifact-relation projection. With no explicit
`--max-new-shards`, ordinary bulk sources transfer at most one new shard per run, GH
Archive transfers at most 48 so a 24-hour daily stream can catch up, and Software
Heritage transfers at most four large origin shards. An explicit budget applies to every
source; zero is valid for a metadata-only smoke test. A healthy budget-limited run is
`partial`, not falsely `complete`.

The enabled source streams have different checkpoint semantics:

- **Hugging Face:** the first scan follows opaque `Link` pagination over the public
  model listing. Later scans start newest-first and stop after crossing the prior
  high-water mark minus the configured overlap. A page budget can leave the source
  partial; the next run resumes its stored URL and frozen cutoff.
- **Framework catalogs:** Torchvision, Keras Applications, Transformers, and Diffusers
  are enumerated from their complete official index documents by declarative structural
  rules. Conditional requests avoid unchanged downloads; a selector that matches no
  entries fails closed rather than tombstoning the prior snapshot.
- **BioImage.IO:** the complete public model-resource index is compared with its prior
  checkpoint. New or changed exact versions fetch their structured artifact manifest and
  version-pinned RDF/model card, verify the index SHA-256, and retain code/config,
  citation, and weight-reference evidence. Removed versions become tombstones only after
  a complete index comparison.
- **OpenAlex:** the default configuration starts with a seven-day publication
  lookback across all domains, ordered oldest-first, then resumes its frozen cursor
  and uses a one-day overlap after each completed watermark. It is a freshness feed,
  not a historical full-corpus scan. The API key is strongly recommended for useful
  production budgets.
- **arXiv, Crossref, Europe PMC, and DataCite:** unfiltered daily windows or OAI
  harvesting retain stable scholarly/deposit identifiers across fields and venues.
  Their complete past is handled by bootstrap or bounded historical windows rather than
  pretending an initial lookback is a corpus census.
- **Semantic Scholar, PubMed, and Common Crawl:** lightweight control adapters enumerate
  complete releases, update files, or WET manifests. The bulk phase then verifies and
  lands only a bounded prefix of exact shards; replay of already committed controls is
  network-free. Each landed WET shard is immediately projected into lossless document,
  contextual-URL, and model-candidate Parquet tables. A separately checkpointed,
  bounded merge admits candidate-bearing documents and exact GitHub relations to search
  and the URL frontier; it can resume a cached shard after interruption.
- **GH Archive:** a no-keyword closed-hour control stream enumerates every public GitHub
  event in each selected hour. The bulk loader verifies and retains every event in
  Parquet; a checkpointed projection deduplicates repository observations by exact
  numeric GitHub ID, enqueues their canonical owner/name URLs, and refuses to advance
  past an incomplete hour. This is an activity plane, not a historical repository
  census.
- **Software Heritage origins:** the control adapter dynamically selects the newest
  settled full-graph release that appears in both the official release document and the
  public S3 inventory, validates its export metadata, and freezes every ORC origin shard.
  The loader streams every binary origin URL into Parquet with exact object and local
  digest evidence. Each committed shard is immediately scanned without name or topic
  filters; structurally exact GitHub repository origins enter the normal metadata/README
  frontier. This covers Software Heritage's archived holdings, not every repository in
  existence, and the origin export contains URLs rather than repository contents.
- **Semantic Scholar projection:** only after all three required datasets are sealed,
  an external-memory hash-bucket join publishes normalized paper/abstract/identifier
  Parquet. Ordered update/delete diffs require a fully verified prior projection and
  preserve tombstone evidence. Ambiguous projection lineages fail closed across process
  restarts rather than selecting an artifact implicitly. A second immutable projection
  retains every extracted candidate's exact span, supporting sentence, identifiers, and
  upstream evidence identity without merging candidates merely because names match.
- **PMC:** the daily OAI-PMH stream retains reusable JATS from the `pmc-open` rights set.
  Its separate bootstrap starts from the repository's declared earliest datestamp.
- **bioRxiv and medRxiv details:** separate first-party streams retain every returned
  preprint version. They query only closed UTC dates, freeze a window while paginating,
  and replay the prior two dates after completion.
- **bioRxiv and medRxiv publications:** separate `pubs` streams preserve delayed
  preprint-to-journal DOI links without merging the two works. They use the same frozen
  closed-date mechanics and replay 90 days because a publication link can arrive well
  after the preprint.
- **OSF community preprints:** the global `osf-preprints` stream keeps every hosted
  preprint community in scope without a provider or subject filter. It uses a frozen
  closed `date_modified` window and restarts page numbering if OSF changes a window's
  declared total.
- **EarthArXiv:** the first-party Janeway OAI-PMH stream retains current EarthArXiv
  preprints with their OAI IDs, DOIs, full-text links, and Dublin Core metadata.
- **HAL:** the first-party, all-record OAI-PMH stream begins at HAL's published lower
  boundary, keeps its date-granularity windows and opaque continuation token stable, and
  retains research-output metadata and links without collection or subject filtering.
- **PLOS:** the publisher's full-document Search API uses publication-date/DOI offset
  windows, preserves its reported total, and restarts a moving window rather than
  advancing through a potentially skipped page. Article/JATS URLs remain non-crawling
  evidence; the source does not retrieve article files.
- **alphaXiv:** this is exact-ID enrichment, never corpus enumeration. The bounded job
  selects arXiv IDs already retained from another active source. Prior per-paper failures
  are persisted and sorted behind previously untried IDs; `--alphaxiv-limit` bounds each
  daily pass, and `--no-alphaxiv` disables it even when a key is present.
- **Artifact relations:** after frontier enrichment, a bounded external-memory join emits
  immutable paper↔code↔model-card↔weights edges. Directed edges preserve contextual URL
  predicates and locators; symmetric edges require an exact artifact identifier or an
  already-resolved model/release identity. Display-name equality never creates an edge.
  Run `modelome link-artifacts` to rebuild this projection independently.

`modelome sync` remains useful for current/control adapters only and runs the URL frontier
after them by default. Disable it
with `--no-frontier`, or run it independently:

```bash
uv run modelome crawl --limit 200 --max-depth 1
```

The crawler accepts only public HTTP(S) targets, enforces robots policy, bounds
response size and depth, enriches GitHub repository links through the API, and
stores other allowed pages as evidence. A direct binary-weight link becomes a
reference-only `weights` artifact without downloading or verifying the file; it is
not by itself a model-release assertion.
Versioned Hugging Face README links are fetched as model cards rather than discarded
as already-enumerated model pages. Successful and exhausted-retry URLs become eligible
for refresh after 24 hours; URLs excluded by crawl policy remain ignored.
GitHub enrichment uses `GITHUB_TOKEN` when configured and is unauthenticated otherwise.
Use a least-privilege token intended for public metadata. Repositories reported as
private are rejected before README retrieval or persistence; merely configuring a token
does not opt the registry into private-source collection.

## Historical API backfills

Daily windows do not fill the past. Run inclusive, bounded windows for OpenAlex, arXiv,
Crossref, Europe PMC, DataCite, bioRxiv/medRxiv details, their publication-link streams,
global OSF community preprints, EarthArXiv, or HAL:

```bash
uv run modelome backfill \
  --source openalex \
  --from 2010-01-01 \
  --to 2010-12-31 \
  --max-pages 100

uv run modelome backfill \
  --source biorxiv \
  --from 2010-01-01 \
  --to 2010-12-31 \
  --max-pages 100

uv run modelome backfill \
  --source datacite \
  --from 2010-01-01 \
  --to 2010-12-31 \
  --max-pages 100

uv run modelome backfill \
  --source medrxiv-publications \
  --from 2020-01-01 \
  --to 2020-12-31 \
  --max-pages 100

uv run modelome backfill \
  --source osf-preprints \
  --from 2018-01-01 \
  --to 2018-12-31 \
  --max-pages 100

uv run modelome backfill \
  --source eartharxiv \
  --from 2017-10-23 \
  --to 2017-12-31 \
  --max-pages 100
```

The default page budget is 100 and must remain finite. The command derives a
durable namespace from the source mechanics and dates, for example
`openalex:backfill:published:2010-01-01:2010-12-31` or
`biorxiv:backfill:2010-01-01:2010-12-31`. Its cursor never overwrites the daily source
checkpoint. Rerun the same command to resume; after upstream exhaustion, another
identical run reports `already_complete` without fetching.

Use `--namespace` only when an orchestration system needs a stable explicit name.
A reused namespace is guarded by a signature of the adapter type, source URL and
mechanics, plus date bounds; changing any of them is rejected. Partition long history
into non-overlapping calendar windows or use each source's official bulk route for very
large loads.

arXiv and PMC expose source-declared earliest boundaries, so they also support complete,
resumable bootstrap without an operator-supplied start date:

```bash
uv run modelome bootstrap --source arxiv --max-pages 1000
uv run modelome bootstrap --source pmc --max-pages 1000
```

`daily` resumes both automatically. Their bootstrap checkpoint namespaces are isolated
from freshness feeds while artifacts retain the same logical source identity, so a
historical replay does not create duplicate papers.

## Provider JSON catalogs

No direct provider endpoint is enabled in `config/sources.toml`. To add one, copy
and adapt the source block in `config/provider-sources.example.toml`; set an
official endpoint, a provider-specific exact-identifier namespace, documented
JSON paths, and the credential environment variable if required. Confirm the
provider's access terms and pagination contract before enabling it.

The generic adapter supports list responses nested at a dotted path, scalar IDs,
names and timestamps, `base_model` values, cursor or same-origin URL pagination,
and model-card URL templates. Authentication is never forwarded to a pagination
URL on another origin. Provider IDs default to mutable model identities. Set
`provider_id_is_release = true` only when the provider documents those IDs as
immutable release identities. The current data model records provider-page artifacts
and models; it does not yet create first-class deployment entities.

## Coverage and benchmark checks

An acceptance manifest is UTF-8 CSV with required `Model` and `Bucket` columns.
Extra columns are allowed. The command exact-matches normalized canonical names
and aliases, and counts a match only when it has active artifact evidence:

```bash
uv run modelome coverage --manifest tests/fixtures/coverage_models.csv
uv run modelome --json coverage --manifest /path/to/independent-acceptance.csv
```

The report includes the manifest hash and observation time, per-bucket recall,
matched model IDs and supporting sources, plus vocabulary-free registry metrics.
It exits 0 when every expectation is found and 1 when any are missing. The manifest
is validation input only; it cannot seed discovery. Treat it as a regression gate,
not as proof of family-wide or global completeness.

Epoch is configured as a live, read-only benchmark rather than a source. List the
available benchmark definitions and evaluate the current store with:

```bash
uv run modelome benchmarks
uv run modelome benchmark --name epoch --minimum-recall 1.0
```

The benchmark command fetches Epoch's current public model corpus, matches it against
current evidence discovered through other sources, and exits nonzero when recall is
below the requested threshold or the live corpus contains invalid rows. It does not
open a write transaction or add benchmark rows to any Parquet dataset. The pre-release
deployment assumes a freshly initialized store, so Epoch data never appears as registry
evidence. Run the benchmark after `modelome daily`, and publish the retrieval time, corpus
size, threshold, misses, and recall with any result.

The included systemd unit does this automatically: a successful daily pass is
followed by the Epoch benchmark at a 1.0 minimum-recall gate. Missing independently
discovered models therefore make the unit fail visibly without adding Epoch rows to
the registry.

## Monitoring and recovery

Use these read-only commands:

```bash
uv run modelome status
uv run modelome stats
uv run modelome dead-letters --limit 100
uv run modelome search "model name"
uv run modelome show MODEL_ID
```

`status` reports each source namespace's checkpoint JSON, completion flag,
upstream count when supplied, cumulative pages/records, and latest run. `stats`
reports aggregate artifacts, active artifacts, revisions, models, releases,
claims, evidence, frontier, conflicts, and dead letters. `dead-letters` shows
quarantined record/fetch errors; it does not automatically retry or delete them.
Search results expose total, current, and active-current artifact support so a model
that survives only as historical evidence is distinguishable from one seen now.

Only one writer process may publish to the Parquet store at a time. Within that writer,
one live sync lease may write a source namespace at a time. A stale run lease is
recovered after its expiry, and claimed frontier URLs likewise return to the queue after
their lease expires. Resuming a checkpoint with changed source mechanics is rejected by
its non-secret configuration signature instead of silently mixing scans.
Malformed individual upstream items are quarantined while valid siblings and the page
artifacts are preserved, but the page checkpoint is retained for retry and the run exits
nonzero. Any frontier fetch errors also produce a nonzero CLI exit; a `partial` frontier
caused only by exhausting its per-run backlog budget remains successful.

Alert on stale `updated_at` values, repeated failed or partial runs, schema errors,
a surprising upstream-count change, identifier conflicts, dead-letter growth, and
pending/failed frontier growth. A `complete` flag means only that the adapter
exhausted its configured cursor or window—not that the source, domain, or global
model population is complete.

Hugging Face, OpenAlex incremental windows, and generic JSON catalogs do not tombstone
records merely because an item is absent from one page or window. Periodic
source-specific full-ID reconciliation remains future operational work.

## Daily scheduling (future corpus operation)

Do not install this scheduler during the entry-first phase. When a future deployment
has explicitly chosen to operate the broader corpus pipeline, the files in
`deploy/systemd` run the one-shot sync at 02:17 UTC each day, add up
to 20 minutes of randomized delay, and catch up after host downtime. Install them
on a persistent host after creating the `modelome` service account and the documented
deployment paths:

```bash
sudo cp deploy/systemd/modelome-sync.service deploy/systemd/modelome-sync.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now modelome-sync.timer
systemctl list-timers modelome-sync.timer
```

The scheduled discovery step runs the complete composed `modelome daily` pass. After a
successful pass, `ExecStartPost` runs the read-only Epoch benchmark. Neither step
ingests the Epoch corpus; a missed benchmark expectation marks the unit failed so it
is visible to service monitoring.

The supplied service expects the checkout and virtual environment at
`/opt/modelome`, optional environment settings in
`/etc/modelome.env`, and writes the searchable Parquet store to `/var/lib/modelome/store` and bulk
Parquet lake to `/var/lib/modelome/lake`.

On another scheduler, invoke the same one-shot command. For example:

```cron
17 2 * * * cd /opt/modelome && .venv/bin/modelome daily
```

Configure a dependent read-only `modelome benchmark --name epoch --minimum-recall 1.0`
job after that pass when reproducing the systemd coverage gate.

For Kubernetes, wrap `modelome daily` in a `CronJob`, mount both `/var/lib/modelome/store` and
`/var/lib/modelome/lake` on durable storage, and prevent overlapping jobs. A `ReadWriteOnce`
volume is the safest default. Multi-writer or object-store deployment is not
implemented; it requires a coordinated publish protocol that preserves the same atomic
commit and evidence contracts.

For an Internet-facing production crawler, route outbound HTTP through an egress proxy
or network policy that denies loopback, private, link-local, and cloud-metadata ranges.
MODELOME validates resolved addresses and every redirect, but the standard-library HTTP
stack performs its own connection-time DNS resolution; application preflight alone is
not a complete defense against DNS rebinding.

## Data stewardship

Raw payloads are evidence, not permission to republish everything upstream. MODELOME
retains the source URL, retrieval/run context, content hash, and original metadata;
license declarations remain part of raw source evidence rather than a fully
normalized license model today. Public exports should expose factual metadata and
short evidence locators unless the upstream license permits more. Preserve
repository-specific licenses, paper/full-text rights, provider terms, robots
directives, rate limits, deletions, and takedown requests. Epoch benchmark rows are
not retained as registry evidence; benchmark reports should still identify and
attribute the evaluated corpus and retrieval date.
