# Architecture

MODELOME is an evidence graph under construction, not a static list of model names.
Its product target is a list of research **entries** that connect a technique, model,
architecture, release, or other research object to its useful resources. The entry-first
operating model is specified in [Entry-first modelome](entry-first.md): ingest one exact
paper, harvest its direct resource links, create or update useful entries, and reuse that
same path when a complete paper source is enumerated later.

The implemented core currently preserves upstream observations, derives a conservative
neural-model projection, admits first-party documented ML-technique assertions, and
supports bounded link enrichment. It is the evidence and identity substrate for the entry
product, not yet a complete entry-materialization or tagging UI/API. Existing large-source
adapters and bulk controls are capabilities to exercise after the small per-paper path is
validated; they are not the current operating schedule. New model names are data; adding
one to runtime code is never a prerequisite for discovery.

MODELOME's central model entities are deep-learning and neural-network architectures,
trained instances, checkpoints, and hybrids with a substantive neural component. A
first-party source can additionally create a **documented ML-technique entry** for a
classical estimator, transformer, or composition component it explicitly publishes; this
is tagged and status-scoped rather than mistaken for a trained model release. This scope
does not narrow acquisition: source adapters still enumerate complete corpora without
method keywords, subject categories, or journal allowlists. Mechanistic models, physical
simulations, and physical/chemical diffusion processes do not qualify merely because
their literature uses the words “model” or “diffusion.”

This document distinguishes the **current Parquet evidence store** from the **target graph**.
That boundary matters: entry and tag materialization, architecture and deployment
entities, fuzzy merge review, bulk-corpus orchestration, and general deletion
reconciliation are design goals, not behavior hidden behind the current CLI.

## Entry-first boundary

There are two different flows, and the system must not confuse them:

| Flow | Current purpose | Input | Output | Scale |
| --- | --- | --- | --- | --- |
| Entry ingestion | Validate and populate useful registry entries | One exact paper identity | Paper evidence, direct resource artifacts, qualified relations, entry update | One paper at a time |
| Corpus enumeration | Later coverage expansion | A source cursor, manifest, or release | Exact paper work items and source checkpoint | Many papers |

Corpus enumeration supplies paper identities; it does not change the semantics of entry
ingestion. In particular, it must not turn into a blanket full-text, repository, data,
or weights download. The current store can retain paper and resource artifact evidence,
but entry-facing records and loose tags remain an explicit next layer. Until that layer
exists, claims that an artifact is connected to an entry are a target contract rather
than a claim about an implemented table.

## Invariants

1. **Evidence before synthesis.** An entry or model assertion and each strong resource
   relation point back to an immutable artifact revision and locator.
2. **Keep the two acquisition modes separate.** A curator may start from one exact
   paper. A later source enumerator must traverse its configured snapshot, cursor, date
   window, or catalog without model-name, subject, or method filters. Expected names are
   allowed only in validation inputs and tests.
3. **Apply scope after acquisition.** The current model projection requires evidence of
   a neural architecture, learned neural parameters, or a substantive neural component.
   Other statistical, computational, and scientific models remain linked artifact
   context rather than false members of that projection. The future entry layer is
   broader: a research object may be useful even before it qualifies as a neural-model
   entity.
4. **Keep evaluation independent.** Benchmark corpora are read-only inputs. Their
   rows never become artifacts, models, aliases, releases, or provenance. Epoch is
   benchmark-only and matches require current evidence discovered by an ingestion
   source.
5. **Keep distinct things distinct.** A paper, repository, model-card repository,
   catalog row, release/checkpoint, provider listing, and entry are related records, not
   interchangeable aliases.
6. **Reuse identity only with exact evidence.** Namespaced identifiers, not name
   similarity, are automatic merge keys.
7. **Commit a page and its checkpoint together.** A restart resumes after the last
   durable page, and replay is idempotent.
8. **Treat disappearance cautiously.** Only a successful authoritative full
   comparison can tombstone a previously observed artifact.
9. **Keep tags loose and separate from evidence.** Tags are optional navigation metadata;
   they do not gate ingestion, prove disciplinary truth, or act as identity keys.
10. **Report bounded coverage.** Cursor exhaustion and benchmark recall describe a
    configured observation, never the unknowable global population.

## Current data model and store format

The implementation persists these layers as logical Parquet datasets in an append-only
commit history. The store is a directory, not a single file. Each successful write
creates an immutable commit under `commits/<sequence>-<digest>-<nonce>/`, containing a manifest
and one Parquet file per logical dataset. `HEAD.json` atomically selects the visible
commit. Readers never observe a partly written snapshot, and an advisory `.write.lock`
permits one writer at a time.

The current v2 manifest records a SHA-256 for every logical table and for every
physical Parquet file. Readers verify the physical files, then verify the state digest
derived from the table-digest map before exposing the commit. This avoids repeatedly
serializing the complete historical provenance corpus merely to validate an unchanged
snapshot. v1 commits remain readable and are validated with their original canonical
whole-state digest until the next write publishes a v2 commit.

### Acquisition and artifacts

- the `source_checkpoints` dataset stores one opaque JSON state document per source namespace,
  completion/upstream-count fields, cumulative page and record counts, and the last
  committed run.
- `sync_runs` stores status, timing, summary counts, and the terminal error for one
  source invocation.
- `artifacts` uses `(source, source_record_id)` as upstream identity and records kind,
  URL, title, source dates, current revision, active state, and tombstone state.
- `artifact_revisions` stores normalized record JSON, raw source metadata, extracted
  text, a content hash, source modification time, run, and ingestion time. The storage
  API preserves these revision rows when it builds every later snapshot.
- artifact identifiers preserve DOI, arXiv, OpenAlex, repository, Hub, or other
  namespaced IDs when an adapter can extract them.

The data model retains source metadata as received; it does not yet have a normalized
rights/license table or external object-store pointer. Public exports must therefore
apply source-specific rights policy rather than assume every raw field is reusable.

The current Parquet schema does **not** yet persist the target product's first-class
`entries`, `entry_artifact_links`, or `entry_tags` tables. The in-progress
`modelome.entries` module does define a portable, offline `Entry` shape and deterministic
exact-ID-first seed builder, including resources and loose tags. `ingest-paper` now
bridges one exact normalized paper observation into the Parquet evidence store, bounded
resource frontier, and a portable entry seed/plan. It does not add first-class entry
tables or materialize entries in the registry; a later explicit corpus build does that.
The store's artifact, identity, provenance, and relation datasets remain the foundation
for that publication layer. Do not conceal the remaining gap by calling a model row an
entry or by storing editorial tags as fabricated source evidence. The target schema and
per-paper contract are described in
[Entry-first modelome](entry-first.md).

### Models and claims

- `models` contains an opaque ID, canonical display name, normalized search name,
  status, and confidence.
- aliases and exact external identifiers are separate tables.
- `artifact_model_links` records which immutable revision asserted a model, including
  local ID, extractor, confidence, locator, and resolution status.
- `model_identifier_claims` preserves both accepted and conflicting identifier claims.
- `model_relation_claims` currently carries relations such as `base_model`; a target
  with a matching exact identifier resolves, while a name-only target can remain an
  unresolved claim.
- `evidence_provenance` stores source-backed predicates and values for model/release
  assertions.

The current extractor is deliberately narrow. It finds candidate names only after
grammatical introduction cues such as “we introduce …” or a distinctive title prefix.
Structured adapters can make higher-confidence model assertions directly. Neither path
contains a production architecture/family allowlist. The target extractor must make a
separate evidence-backed scope decision so corpus-wide discovery does not turn every
named statistical method, mechanistic model, or scientific diffusion process into a
registry model.
Declared identifiers, links, models, releases, and relations carry `source-declared`
provenance; derived candidates carry the extractor's own versioned name. Reprocessing an
unchanged revision replaces that extractor's current projection, so a withdrawn derived
candidate no longer counts as current evidence while the immutable source revision stays
intact.

Complete Semantic Scholar releases take the same extraction path at corpus scale. A
verified external-memory paper/abstract/identifier join is streamed into an immutable
model-candidate Parquet projection. Each row keeps the exact source span, supporting
sentence, extractor identity, upstream identifiers, and source-projection artifact ID.
These rows are assertions, not name-based merge decisions; integrating them into the
unified cross-source searchable entity projection remains separate work.

### Releases and checkpoints

`model_releases` is first-class in the current schema. A `ReleaseHint` must refer to a
model asserted by the same source record and must supply a version, revision, or exact
external identifier. Release identifiers are resolved separately from model identifiers,
and conflicts remain source-backed claims. Artifact/release links retain version,
revision, release time, metadata, confidence, locator, and extractor.

The Hugging Face adapter emits one release assertion per repository observation. A
commit SHA becomes an exact `huggingface:revision` identifier; when SHA is absent,
`main` is descriptive rather than a global exact identity. Weight filenames are release
metadata, not downloaded or content-verified weight objects. The generic provider
adapter emits releases only when its operator explicitly declares provider IDs immutable.

### URL frontier and cross-source evidence

Every explicit source link is evidence. Crawlable public links enter a depth-bounded
frontier; URLs merely found inside arbitrary nested raw metadata remain in the immutable
raw evidence rather than automatically becoming fetch work. GitHub repository URLs use
a dedicated metadata/README fetcher. Other allowed HTML/text pages use a generic web
fetcher, while a versioned Hub README is materialized as model-card text. A directly
linked binary weight becomes a reference-only artifact; the binary, PDFs,
already-enumerated OpenAlex pages, and rendered Hub model pages are not recursively
downloaded by the frontier. Finished or exhausted-retry fetches are eligible for a
bounded refresh after one day; policy-ignored URLs remain terminal.

`show` connects supporting artifacts when they share an exact artifact identifier, a
normalized canonical URL, or a retained requested-URL alias after redirect
canonicalization. This is evidence navigation, not an automatic assertion that two
differently identified models are identical.

The separately sealed artifact-relation projection materializes these connections for
corpus-scale consumers. It emits directed contextual URL edges such as
`official_implementation`, `implementation`, `weights`, and `model_card`, plus symmetric
connections backed by exact artifact identifiers or already-resolved model/release IDs.
It scans only active artifacts and current revisions, uses bounded hash partitions, and
retains the evidence locator and both revision identities. Name equality is never a join
key. Rebuilding after a correction removes stale derived edges while older immutable
projections remain auditable.

### Unified entry construction

`modelome.entries` provides the implemented, opt-in entry-construction contract.
It is deliberately separate from the evidence store and has no implicit corpus scan:
a source adapter can serialize one `SourceRecord` into an entry seed, then a worker
can resolve only that seed's declared URLs. A later finite-seed build produces one
resource-complete entry per exact-identifier component. Tags are arbitrary strings
with retained seed provenance and are never entity-merge keys. Source-declared model
lineage is retained on the subject entry and resolves to another entry only through a
same-record local ID or an exact model identifier; it is never a merge key. See
[entry construction](entries.md) for the executable commands and input contract.

## Exact-ID-first resolution

The implemented resolver is intentionally stricter than a conventional entity matcher:

1. Normalize a source-supplied identifier namespace and value.
2. Reuse a model or release only when an existing exact identifier resolves to one
   compatible entity.
3. Preserve conflicting exact-identifier claims instead of overwriting either side.
4. Give identifier-free hints deterministic artifact-local identities.
5. Add names and aliases for search, never as an automatic cross-artifact merge key.

Consequently, the same spelling in two independent papers can remain two models until
an exact link or a future reviewed resolution decision connects them. This favors false
splits over irreversible false merges.

The target resolver can later add explicit cross-references, blocked fuzzy candidates,
scored merge proposals, human decisions, and reversible splits. None of those future
stages should silently weaken exact identifier scope: a paper DOI identifies a paper,
a repository ID identifies a repository, and a provider alias does not prove an
immutable checkpoint.

## Discovery flow

```text
exact paper seed -> normalize -> store revision -> extract declared resources
                                               |
                                               +-> record/enqueue bounded direct links
                                               +-> resolve exact IDs and relations
                                               +-> materialize an entry (target layer)

later source enumerator -> durable exact-paper worklist -> same flow above
```

The first line is the current product workflow. The codebase's enabled enumerators include cursor/high-water APIs, complete static catalogs,
closed-date feeds, OAI-PMH repositories, global public-note traversal, complete bulk
control planes, and an unfiltered public-GitHub activity stream. BioImage.IO supplies an
authoritative exact-version model/package plane; Hugging Face, OpenRouter's current provider
catalog, and the framework catalogs
supply independent code, card, release, and documentation planes. GH Archive supplies
exact repository observations from activity, and the GitHub frontier enriches them with
current metadata and bounded README text. Software Heritage adds a historical origin
plane: every URL from the selected settled full-graph origin export is retained in
Parquet, while only structurally exact GitHub repository origins enter that same frontier.
ModelScope adds a separate, explicitly bounded public model-card plane: its official API
caps anonymous pagination at 3,000 records per documented sort, so each sweep records only
the configured sort windows and never takes authoritative-snapshot deletion semantics.
NVIDIA NGC adds a guest-visible current `MODEL` catalog plane: its public search endpoint
is checked for stable totals during one deterministic, paginated sweep, but it likewise
never takes deletion semantics because it is a moving provider listing rather than a
versioned export. Each row preserves an NGC-native model ID, latest-version handle, and
first-party model-card page. Its exact-version metadata endpoint can subsequently resolve
one entry's declared model-card Markdown, paper/code/documentation links, and release data;
the catalog enumeration itself still does not fan out to cards, archives, or weight bytes.
fastText contributes a first-party static-vector release plane: every structurally matched
Common Crawl/Wikipedia language row keeps its native language handle, binary/text artifact
references scoped to that one release, and the page's directly declared paper, implementation,
and license links. The currently observed page contains 158 matching rows despite describing
itself as “157 languages”; this is retained as a source discrepancy rather than normalized away.
GluonCV contributes historical pretrained classification, detection, segmentation, pose,
action-recognition, and depth planes from its maintained 0.11 Model Zoo: each structural
table row keeps its exact model handle, and any row-declared training recipe/log/config stays
scoped to that model rather than being copied to its siblings.
Ultralytics adds an independent current/historical vision-family plane for documented YOLO,
SAM, RT-DETR, and related releases; every table row retains the source's canonical model-page
link without turning a decorative release badge into part of the model identity.
PaddleNLP contributes its first-party transformer-family task matrix: every structural model
row retains the native family identifier, while the row's directly declared arXiv reference,
when present, is scoped to that family. The matrix is documentation evidence, not an assertion
that every family has a checkpoint or a complete release history.
Its separate pretrained-embedding matrix retains every exact, loadable Word2Vec, GloVe, and
fastText identifier while explicitly discarding the publisher's `N/A` cells; because the page
does not declare stable per-artifact download URLs, each entry carries the source document rather
than a fabricated artifact reference.
Sentence Transformers contributes its maintained curated pretrained-model document with each
selected Hugging Face owner/repository path represented as the same exact `huggingface:model`
identifier the Hub enumeration uses. The document's presentation text is not treated as identity.
spaCy contributes its maintained compatibility manifest as a historical language-pipeline plane:
each package has a `spacy:model` identifier and each published package version is retained as a
separate exact release observation. The manifest does not establish a stable wheel URL, so no
binary reference is synthesized.
Stanza contributes every retained public resources manifest at one immutable source revision.
Each language/processor/package coordinate is an exact `stanza:model`, while every manifest
entry with a declared MD5 is a release observation; processor dependencies are retained as
non-identity edges. The resource manifests do not themselves declare individual artifact URLs,
so checksums are preserved without fabricating download links.
Ollama's public library is a separate conditional-GET document snapshot whose card slugs and
declared titles become provider-native model-family evidence without fetching manifests or
model blobs. Cloudflare Workers AI contributes one separate current provider catalog: each
listed card exposes its native runtime ID, display label, and documentation URL, but does not
assert source-artifact provenance or historical completeness. Zenodo adds a broad, exact
resource-type Model catalog with record/DOI/version/related-resource/file evidence, while
keeping its cross-domain classification distinct from a neural-model claim. Every landed Common Crawl WET shard is also converted to immutable
document/URL/candidate evidence and streamed into search only when it contains a model
candidate or exact GitHub relation. OpenRouter contributes one complete public current
provider-availability catalog, retaining its exact model IDs, aliases, and declared Hugging
Face links without treating them as identity merges. The optional OpenAI Models API adapter records
only configured public-owner IDs from an operator-authorized account response; unrecognized owner
rows are counted but never persisted, and availability never becomes a weight or provenance claim.
Anthropic's optional catalog follows its source-defined API-version-header and cursor contract,
retaining available model IDs and capability metadata without treating them as artifact evidence.
The timm source registry resolves one public
commit and statically parses the model modules imported by its own package initializer, preserving
registered architecture aliases and exact declared configuration/Hub/weight links without executing
upstream code. PaddlePaddle ModelCenter independently supplies source-native family and download
table release records at one public repository commit, including only explicit documentation, project,
paper, and artifact links. ESPnet contributes its maintained speech Model Zoo table, preserving
task/corpus metadata and provider-invalidated records rather than dropping historical releases.
Fairseq's retained wav2vec, wav2vec 2.0, and neural-machine-translation tables are independent
pinned document planes: each reader accepts only the configured source header plus same-row direct
artifact, so the original and 2.0 speech catalogs never get conflated merely because they share a
README file or a similar method name.
Omnilingual ASR's source-controlled cards use a deliberately non-general parser: only literal
top-level named cards with direct checkpoint URLs can create releases, while tokenizer and incomplete
cards are ignored. Seamless preserves an adjacent multilingual release plane from source tables whose
rows directly declare model-card, checkpoint, and metrics links; metrics retain evaluation evidence
rather than acquiring a weight relation solely because the archive happens to be a ZIP file.
SONAR's text-model table is similarly constrained to its exact encoder/decoder release labels, leaving
the tokenizer and the much broader language-support tables as non-model source evidence.
Original Segment Anything's three explicit backbone checkpoints use an even narrower headed-list
contract: every selected bullet has exactly one source-native `vit_*` handle and exactly one direct
checkpoint, while the rest of the README is outside its ingestion scope.
BEiT and BEiT v2 use named task-result rows, but their source-declared initialized-checkpoint
field is the stable release context: rows sharing the same model/checkpoint are unified across
task headings and retain each direct fine-tuned weight or training-log resource. When authors call
every resource merely `link`, the row's explicit `weight` or `log` header supplies the relation;
a named source label still takes precedence. BEiT-3 instead uses the narrower `Download Checkpoints`
list contract, which admits only its six named model checkpoints and excludes the adjacent tokenizer
and fine-tuning tables. None of those README-level paper mentions is broadcast to release rows.
Barlow Twins and SwAV use direct historical release tables. I-JEPA's equivalent HTML table has
repeated architecture names, so its source-declared patch size, resolution, training duration, and
corpus fields form a single exact release context. This avoids collapsing separate ViT-H releases
while retaining their direct checkpoints, logs, and configuration references as row-local resources.
DiT's table independently preserves the named XL/2 architecture and its adjacent released image
resolution, so the 256 and 512 checkpoint URLs remain distinct first-party release records.
AudioCraft's MusicGen/MusicGen-Style/AudioGen, MAGNeT, JASCO, EnCodec, and AudioSeal documents
expose exact public Hugging Face model-card URLs instead of direct binary links. Their constrained
section readers retain those card identities (including MusicGen stereo and Audio-MAGNeT variants),
add the URL-proven `huggingface:model` bridge, and exclude the pages' demos, prose, and generic
documentation links; they never claim that a card URL is a checkpoint download.
DETR and ConvNeXt keep their release tables separate from one another and retain the first declared
backbone or resolution field as row context. This prevents source-local display-name reuse from
collapsing an R50/R101 or 224/384 checkpoint; DETR training logs remain attached as resources rather
than being mistaken for weight files.
Swin Transformer keeps the same release-table rule for its V1, V2, and SwinMLP rows, retaining the
publisher's pretraining field so ImageNet-1K and ImageNet-22K observations do not conflate.
DeiT, CaiT, ResMLP, and PatchConvNet are four retained documents in one repository, but each has a
separate source namespace and pinned document cursor. CaiT and PatchConvNet use the same resolution
context rule where necessary; shared repository prose and title similarity cannot merge identities.
PySlowFast and TimeSformer add independent video-model release planes. PySlowFast preserves the
publisher's architecture/size context, while TimeSformer retains its declared training dataset; direct
`.pkl` checkpoints are classified as weights without relying on an ambiguous link label.
PaddleClas's public command-line registry complements ModelCenter with literal ImageNet-series and
PULC inference names paired with exact archive templates; the reader AST-parses those declarations
at one commit and never imports upstream code or fetches an archive.
PaddleDetection's versioned config README model-zoo tables add direct checkpoint rows through one
bounded source-archive read, preserving every row's model-card page and explicit pinned config links
without interpreting code or transferring checkpoint bytes.
Paddle3D applies the same bounded static reader only to its first-party per-architecture
`docs/models/**/README.md` pages; each admitted checkpoint must have an explicit model-zoo table
row and direct recognized artifact URL, while the broad root architecture menu is not promoted into
release evidence.
PaddleGAN's English tutorial archive is a separate generative-vision release plane: only direct
first-party files under its declared model-download path are admitted. Table resources remain row
local, and document-level paper/code/example links are attached only to a tutorial that declares
exactly one such artifact, so multi-model pages cannot broadcast a generic reference.
The same bounded static reader separately covers PaddleSeg's config model-zoo tables under a distinct
identity namespace; compact Markdown table separators and escaped row-cell delimiters are parsed
structurally rather than discarded as malformed documentation.
PaddleVideo uses individual English model-zoo Markdown pages instead of config READMEs; the source
config narrows the same reader to that directory and `.md` suffix while retaining only table-declared
checkpoint URLs under a separate namespace.
PaddleSlim's maintained English Model Zoo provides a distinct table-release plane for compression,
distillation, pruning, quantization, and NAS models. It retains only named rows with direct
source-declared downloads and keeps them separate from the upstream models they may compress.
PaddleRec's exact `Type / Algorithm / Paper` support table supplies a neighboring documented-technique
plane: its implementation, documentation, hosted-example, and paper links are preserved on the named
recommendation algorithm's row, without pretending that the table declares a trained checkpoint.
Reusable first-party table ingestion covers PaddleSpeech's released neural models, PaddleOCR's V2
releases, and PaddleOCR's current HTML model tables, with explicit same-row artifact admission and
no name-list scraping. Detectron2
resolves its static, first-party config/checkpoint map at one commit and AST-parses the map without
executing project code, retaining exact config/source/repository/weight references. A token-gated
Replicate adapter adds displayworthy public model cards, direct resources, and checkpointed
per-model version history
when an operator explicitly provisions its public-model token. The optional OpenAI adapter uses an
operator-provisioned key but filters out non-public-owner account rows before they can persist, leaving
only source-scoped provider availability evidence. The generic JSON catalog adapter carries only
validated noncredential static headers, a separately held credential header or query parameter,
bounded response bodies, and explicit cursor-completion conditions for provider APIs such as
Anthropic's and Gemini's. It remains
available for other operator-configured provider endpoints. The URL frontier
enriches links discovered by those primary sources; it is not by itself a general-purpose
Internet crawler. Neither the activity stream nor Software Heritage's archived holdings
prove an exhaustive historical inventory of every dormant or deleted repository.

OpenMMLab's structured model-index plane includes both active projects and retained historical
toolboxes: final MMClassification 0.x, MMTracking, MMSelfSup, MMFewShot, MMFlow, and MMGeneration
each have their own immutable revision cursor. The same source reader traverses only imported model
metafiles and keeps their explicit configuration, weight, code, README, and paper declarations local
to each project record; similarly named models are never merged across projects merely because they
share a label.
MMEngine's source-controlled OpenMMLab and MMClassification JSON maps add a complementary
pretraining-release plane: each literal runtime handle is admitted only with its direct checkpoint
file URL, pinned source map, and source implementation. The reader does not import MMEngine, infer
architectural equivalence, or treat the package's loading mechanism as evidence beyond that map.
The same constrained model-map contract covers OpenAI CLIP and Whisper: one named literal Python
dictionary is AST-parsed at a pinned commit, and each handle retains only its direct checkpoint,
source file, and source implementation. The parser rejects dynamic expressions, duplicate keys, or
non-checkpoint values; upstream Python is never imported or run.
The first-party MAE checkpoint table uses the inverse visual layout: named ViT variants are column
headers and one explicit `pre-trained checkpoint` row supplies direct weights. The table reader can
transpose only a configured matching row, retaining each header/cell pair independently and refusing
the surrounding metric and training-prose cells as model evidence.
SAM 2 and ImageBind use the ordinary row-oriented form. SAM 2 retains only its source-declared Hiera
model prefix from an otherwise presentational model cell, while ImageBind retains its named model row;
both still require a same-row direct checkpoint and do not broadcast their page-level paper link to
individual release records.
MMHuman3D predates that index convention, so its independent archive reader scans only the first-party
per-method `configs/**/README.md` release tables. It accepts direct recognized checkpoint URLs and
their linked Python configurations, never promoting the broader supported-method menu into releases.
The retained original MMAction project is another legacy plane: an exact root `MODEL_ZOO.md` selector
admits only its source-declared checkpoint rows, avoiding repository-wide scans and keeping its 2019–20
action records independent from the newer MMAction2 manifest source.
MMFashion's first-party historical Model Zoo is read as one pinned Markdown table document. Its
Backbone/Model-type rows require a direct declared artifact; Google/Baidu hosts remain reference-only
resources and task-table context never becomes a guessed release identity.
OpenUnReID adds an independent archived re-identification plane. Its visible Method/Download tables
are source evidence, while HTML-comment examples are removed before table parsing without changing
line locators; source-relative method links supply labels but are never mistaken for artifacts.
TensorFlow TPU, CenterNet, CenterTrack, AlphaPose, pycls, VISSL, MaskFormer, Mask2Former, legacy Detectron, detrex, Detic, and X-AnyLabeling add separate first-party historical
release-table planes. Their compact document readers require both the explicitly headed model column
  and a direct same-row artifact; table headings, source document, and repository remain provenance
only, with no claim that matching model labels across projects denote a single identity.

Epoch is deliberately outside this discovery flow. Its current public model corpus is
fetched only by the read-only benchmark command and compared with independently
discovered registry evidence. It is never passed through the ingestion path or
committed as source data. The pre-release store format assumes a freshly initialized
store, which therefore contains no Epoch artifacts or evidence.

This source breadth is a later corpus-scale capability inventory, not a requirement for
the entry-first phase. Framework registries,
additional paper indexes, domain repositories, provider-specific endpoints, software
archives, and historical web corpora require additional adapters or configurations.
Complete Semantic Scholar datasets form the planned Priority 0 global paper plane. The
enabled direct bioRxiv and medRxiv streams plus planned PubMed, PMC, and Europe PMC work
form its biomedical counterpart. These are first-class enumerators, not optional
enrichments behind OpenAlex.

## Incremental commits and recovery

For each page, the sync engine:

1. acquires the store's advisory writer lock and the source namespace's renewable run
   lease;
2. fetches a page using its opaque checkpoint state;
3. applies valid records to an isolated in-memory snapshot;
4. quarantines validation/persistence failures as dead letters;
5. writes complete logical Parquet datasets and their manifest into a staged commit
   directory, fsyncs and renames that directory, then atomically replaces `HEAD.json`;
6. advances the source checkpoint only when no item was quarantined; and
7. stops on upstream completion or the per-source page budget.

The same `(source, source_record_id, content_hash)` observation is idempotent. A run can
be `partial` because it hit its page budget, `complete` because it exhausted its current
cursor/window, or `failed`. “Complete” is always source/query scoped.
If any item is quarantined, valid siblings remain idempotently stored but the cursor stays
at that page and the run fails, so a corrected adapter or upstream item can be replayed.

Timestamp adapters overlap their previous high-water mark to tolerate late changes and
clock/pagination imperfections. Future authoritative snapshot adapters must validate a
complete comparison before treating absence as deletion evidence; partial windows and
failed records cannot imply tombstones.

Each stored cursor carries a non-secret signature of the adapter mechanics that define
its traversal. A changed endpoint, mapping, filter, or pagination contract cannot resume
an incompatible cursor accidentally. Expired run and frontier-claim leases are
recoverable after interruption.

The first implementation writes a complete logical snapshot for every commit. This
keeps reads simple and makes every published state self-contained, but old commit
directories consume space as the registry grows. A future compactor will run under the
same writer lock, materialize a fresh self-contained checkpoint, verify its manifest,
publish it through the same atomic `HEAD.json` swap, and then garbage-collect only
unreferenced physical commits according to a configured retention policy. Compaction
must preserve all logical artifact revisions and provenance. No automatic compactor is
implemented yet; operators should not manually remove the commit selected by
`HEAD.json`.

## Historical backfill and reconciliation

`modelome backfill` implements inclusive windows for OpenAlex, bioRxiv/medRxiv details, and
their independent publication-link streams. Each window uses a source namespace
separate from daily ingestion, stores the upstream cursor, freezes its boundaries,
validates a source/config signature on resume, and becomes a no-op after completion.
Operators can partition history into bounded non-overlapping windows.

These are API backfills, not the official OpenAlex snapshot or bioRxiv/medRxiv monthly
TDM loaders. The target system should prefer bulk snapshots for very large histories,
partition snapshot shards, meet incremental ingestion at a recorded watermark, and
reconcile record manifests.

General reconciliation is not implemented today. Future source-specific jobs should:

- compare local IDs/counts with authoritative manifests;
- repair missing or corrupt partitions without resetting a good daily cursor;
- detect redirects, repository moves, changed license declarations, and provider aliases;
- tombstone only after a complete trustworthy comparison;
- replay extraction/resolution when their version changes;
- sample conflicts, apparent duplicates, and unresolved clusters.

## Coverage and quality

`stats`, `status`, `coverage`, and `benchmark` report different bounded facts:

- `status` exposes checkpoint state, source completion, upstream count when available,
  cumulative pages/records, and the latest run.
- `stats` provides aggregate dataset, conflict, active-artifact, and frontier counts.
- `coverage --manifest` exact-matches an external `Model,Bucket` acceptance corpus against
  canonical names/aliases with active support from a current artifact revision. It
  reports per-bucket recall,
  matched IDs/sources, manifest hash/time, source counts/completion, exact-identifier
  model count, model statuses, relation state, and frontier state.
- `benchmarks` lists configured external benchmarks. `benchmark --name epoch` fetches
  the live Epoch corpus and measures recall using only independently sourced current
  evidence. Benchmark rows are not persisted. `--minimum-recall` makes the command exit
  nonzero below the requested gate.

The bundled acceptance fixture spans classic vision, GAN, transformer, image diffusion,
scientific diffusion, and neural models in biology, medicine, chemistry, physics, and
materials. Passing it proves only that those
examples are currently observable under the inspected store and matching rule. It
does not prove that all GANs, transformers, diffusion models, scientific domains, or
upstream records have been indexed. Manifest bucket and reference columns are reporting
metadata only; they do not classify or seed registry entities.

Future quality work should add independently sampled extraction precision/recall,
resolution error rates, domain/year/language distributions, source-overlap estimates,
lineage completeness, in-scope/out-of-scope confusion matrices, and versioned evaluation
reports outside the discovery store. Domain suites must include hard negatives such as
MCMC-only analyses, mechanistic simulations, and papers about physical diffusion as well
as neural hybrids that must remain in scope.

## Target graph extensions

The longer-term graph should add first-class entities and audited decisions without
discarding current evidence:

- **entry:** the user-facing research object, distinct from any paper, repository, or
  release, with a canonical display name, aliases, status, and supporting resource
  relations;
- **entry tag:** optional, loose editorial navigation metadata, separate from source
  evidence and never used as an ingestion filter or exact identity key;
- **architecture:** structural design distinct from a trained model or family;
- **deployment:** dated provider/region/endpoint availability distinct from an immutable
  release;
- **training run and package variants:** explicit fine-tune, adapter, quantization,
  precision, and component lineage;
- **review decisions:** versioned merge, split, contradiction, and exclusion outcomes;
- **rights projections:** normalized license observations and export policy;
- **distributed persistence:** a coordinated multi-writer commit protocol plus permitted
  raw object storage, preserving the current logical identity and revision contracts.

Until those entities exist, provider catalog rows are provider-page artifacts linked to
models, not deployments; model names should not be interpreted as architectures; and a
weight filename is not a verified tensor digest.

The retained architecture, component, ancestry, and evidence relations should be
sufficient for a future phylogeny analysis. That analysis is a downstream consumer, not
an ingestion taxonomy or a reason to place non-neural methods into the model population.

## Why literal completeness is unprovable

There is no closed population called “all deep-learning models.” Architecture, family,
training run, checkpoint, adapter, quantization, ensemble, and API alias can each be
called a model. New instances appear continuously, while many are private, unnamed,
deleted, never documented, inaccessible to automated retrieval, or described with
domain language that does not resemble familiar deep-learning terminology.

MODELOME can make a narrower, auditable statement at a dated snapshot: it enumerated named
sources under documented access rules, processed a reported portion of their configured
collections/windows, achieved measured recall on an independent sentinel corpus, and
retained revision-level evidence for indexed assertions. That is a testable path toward
universality without claiming an impossible proof of global completeness.
