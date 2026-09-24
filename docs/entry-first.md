# Entry-first modelome

## Decision

MODELOME is a registry of **research entries**, not a paper warehouse. An entry is a
single navigable record for a technique, model, architecture, trained release, or
other research object worth following. It gathers the authoritative and useful
resources around that object: papers, preprints, repositories, model cards, weights,
documentation, benchmark pages, datasets, demos, and other relevant links.

The immediate operating mode is **one paper at a time**:

1. start with one exact paper identity;
2. retrieve the metadata and text that may lawfully be used for that paper;
3. collect its directly supported resource links;
4. write or update the corresponding entry with those links and their provenance; and
5. move to the next paper.

We are deliberately *not* trying to download every paper, scrape every site, or
materialize a giant full-text dataset now. Those are later scale problems. The work now
is to make the ingestion path correct, restartable, and pleasant enough that a single
paper can reliably turn into a useful entry. Once that path is solid, the same scripts
can be driven by a complete paper enumerator.

This document is the product-level source of truth for that distinction. The existing
Parquet evidence store, source adapters, bulk loaders, and frontier crawler are useful
building blocks; they are not a mandate to run a global harvest during this phase.

## What an entry is

An entry answers a practical question: “I have heard of this thing. Where are the
resources that let me understand, reproduce, evaluate, or use it?” It should be useful
even when it contains only one paper and one official repository.

An entry is **not**:

- a copy of every document associated with a topic;
- a claim that all linked resources are interchangeable or maintained by the same
  people;
- a bibliographic database record with an arbitrary list of URLs; or
- a closed ontology that decides in advance what every field must call the thing.

The target entry is a small evidence graph with a human-readable presentation. The
entry has a canonical display name and may have aliases, but its real substance is the
set of source-backed resource and relation claims below it.

```text
entry: “Example method”
  │
  ├── described_by ── paper / preprint
  ├── implemented_by ── code repository
  ├── released_as ── model card / checkpoint
  ├── evaluated_by ── benchmark or leaderboard
  ├── uses ── dataset / dependency / base model
  └── documented_by ── project page / docs / tutorial

Each arrow records: source, locator, observed time, confidence, and any caveat.
```

The visible entry can group links into friendly sections. The stored graph must retain
the exact relation and the evidence that justified it, so a user can distinguish an
official implementation from an author's related repository or a third-party tutorial.

## What belongs in an entry

The registry should preserve a resource as an artifact rather than flatten it into a
single URL field. Important artifact kinds include:

| Resource | Typical stable identity | Example relation to the entry | Notes |
| --- | --- | --- | --- |
| Paper or journal article | DOI, OpenAlex, publisher record | `described_by` | A DOI identifies the paper, not the technique. |
| Preprint or version | arXiv ID/version, bioRxiv DOI/version | `preprint_of`, `described_by` | Preserve version identity; do not overwrite it with a journal article. |
| Code | Forge repository ID/URL and commit when known | `official_implementation`, `implementation` | “Official” requires source evidence, not an inference from the name. |
| Model card or package | Hub repository and revision | `model_card`, `released_as` | A card and a checkpoint are related but different artifacts. |
| Weights | Provider/repository revision and file URL | `weights` | Retain a reference by default; do not download multi-gigabyte binaries. |
| Dataset | DOI, registry ID, or landing URL | `trained_on`, `evaluated_on`, `uses` | Record the asserted relation and its source. |
| Benchmark / leaderboard | canonical page URL | `evaluated_by` | A score page is not proof of authorship or release status. |
| Documentation / demo / project page | canonical URL | `documented_by`, `demo` | Useful resources can be first-class links without being model identity evidence. |

The table is deliberately extensible. “Whatever” is a valid future category when a
resource is helpful and its relationship can be described honestly. Do not add a new
entity class or schema migration merely because a new kind of useful link appears;
start with an artifact plus a typed relation. Promote it to a first-class concept only
when its lifecycle, identity, or queries demonstrably require that.

## Identity: a useful entry without premature merging

Papers, source code, checkpoints, and names identify different things. MODELOME must
not merge them merely because they share a title, acronym, or author list.

The safe rules are:

1. An exact, namespaced identifier may join the *same kind* of artifact across sources.
   For example, an arXiv identifier supplied by both OpenAlex and arXiv can connect two
   records about the same preprint.
2. A paper's DOI may support an entry, but it does not by itself name the entry. A
   paper may introduce several techniques, and a technique may be described by many
   papers.
3. A link becomes `official_implementation`, `weights`, `base_model`, or another
   strong relation only when the paper, source provider, or another retained artifact
   explicitly supports that interpretation. Otherwise use a weaker relation such as
   `implementation`, `references`, or `related_resource` and show the caveat.
4. Same-name entries stay separate by default. A future review decision may merge or
   split them, but similarity search must never silently rewrite identity.
5. Every derived link retains the source artifact revision and a locator: a field,
   section, sentence span, table cell, or supplied API field where practical.

False splits are inexpensive to review. False merges destroy the entry's usefulness,
especially in biology where the same short name frequently labels a method, a dataset,
and a software package.

## Tags: loose navigation, not an ontology gate

Entries need a light way to browse across departments and fields. Tags are that layer.
They are intentionally loose, additive, and revisable. Examples include `bioML`,
`protein design`, `single-cell`, `medical physiology`, `computer vision`, `chemistry`,
or `diffusion`.

Tags have these constraints:

- They support browsing, filtering, and future editorial organization. They do not
  determine whether an artifact can be ingested.
- There is no required hierarchy, controlled vocabulary, or promise that two nearby
  tags are semantically equivalent. A tag can be a department, field, modality,
  organism, use case, or broad technical theme.
- Preserve the spelling a curator supplied, while a presentation layer may offer
  case-insensitive search and suggested synonyms. Do not force a curator through a
  taxonomy before they can save an entry.
- Tags must be separable from evidence. A paper can support the claim that a method
  exists; it usually does not prove that a curator's broad tag is the one correct
  disciplinary home.
- The absence of a tag means “not tagged yet,” not “outside this field.”

A minimal entry therefore needs only a title, at least one evidence-backed resource,
and zero or more tags. Descriptions, aliases, summaries, and relationship notes improve
it but should not block ingestion.

## The single-paper ingestion loop

The unit of work is an exact seed paper, supplied as a DOI, arXiv identifier, PMID,
PMCID, OpenAlex work ID, source URL, or an equivalent stable source identity. A title
alone is a search aid, not a safe seed when a stable identifier is available.

### 1. Resolve the paper record

Retrieve source metadata from the source that owns the supplied identity, then use
cross-source identifiers it declares to find corroborating records. Persist:

- the input identity and normalized identities;
- title, authors, venue, publication/version dates, abstract when allowed, and source
  URLs;
- the source response or normalized evidence needed to audit the mapping;
- access and license observations; and
- the exact source revision or retrieval time.

The process should prefer open metadata and author-hosted/open-access text. An
institutionally authorized local text import is allowed only when the operator has
already obtained the text under an applicable license. It must not automate a browser
session, reuse credentials, or turn licensed publisher content into a public export.

### 2. Create a provisional entry

The paper may describe no entry, one entry, or several. Create a provisional entry only
when there is retained evidence for a particular technique/model/object. If the paper
does not introduce a named object, keep it as an artifact and its links; do not invent a
new entry just to satisfy a one-paper-one-entry rule.

For a newly introduced object, the paper title and introduction are common starting
evidence. The initial entry may be marked `candidate` or `documented`, depending on the
strength of that evidence. A source-declared model card or release can supply stronger
status later.

### 3. Harvest direct resource links

Collect only links that the paper or its authoritative metadata directly exposes first:

- publisher/preprint landing pages and alternate versions;
- code, project, documentation, model-card, and demo links;
- dataset and benchmark links explicitly named in a resource section, table, or
  structured metadata;
- model/package/release identifiers and linked weight references; and
- correction, retraction, or “published as” links.

For every link, record the URL, its normalized canonical form after a safe redirect,
the relation candidate, the evidence locator, and whether it is suitable for a bounded
follow-up fetch. The default depth is one hop from the seed paper. A fetched repository
or model card may reveal its own direct release links, but that does not authorize an
unbounded recursive crawl.

### 4. Enrich each admitted primary resource

Fetch only the small, public, policy-permitted metadata or text needed to characterize
the resource. Respect robots policy, rate limits, provider terms, and content-size
bounds. Do not download arbitrary PDFs, repositories, datasets, or weight files merely
because they are linked.

Use the enrichment to make relationships more precise. For example, a repository found
in a paper may remain an `implementation` until its README or the paper explicitly
calls it official. A direct binary-weight URL is stored as a reference-only weights
artifact; fetching its bytes is a separate, explicit future workflow.

### 5. Resolve exact identities and attach relations

Deduplicate artifacts by their exact identifiers or canonical URLs. Attach source-backed
relations to the provisional entry. Preserve conflicts instead of picking a winner
silently. A link without enough support remains in the entry as a qualified resource,
not as a fabricated strong claim.

### 6. Add lightweight tags and publish

Add any useful loose tags. They are editorial navigation metadata, so they may be
changed later without revising the underlying artifact evidence. Publish the entry when
it is minimally useful, then queue missing enrichment rather than holding it hostage to
perfect completeness.

### Definition of done for one paper

One paper is successfully ingested when:

- its stable source identity and provenance are stored;
- the system has recorded the paper's direct, supported resource links;
- each retained link has an honest relation or is explicitly marked unresolved;
- every new entry has at least one supporting artifact and a clear status;
- resource identities are exact-ID/canonical-URL resolved where possible;
- the operation can be rerun without duplicating artifacts or relations; and
- failures are visible and retryable without losing the already stored evidence.

It is *not* a failure if no code, weights, or dataset exists, if some pages are private,
or if a paper creates no named entry. The durable output is the audited resource graph,
not a forced catalog card.

## Script architecture: now and later

Build the scripts as separate, composable stages. A later global run should call the
same paper-ingestion stage that a curator uses today.

```text
exact paper seed                later complete source enumerator
        │                                   │
        └───────────────┬───────────────────┘
                        v
              resolve / normalize paper identity
                        v
              store immutable paper observation
                        v
          extract declared resources and relation evidence
                        v
        bounded enrichment of admitted primary resource links
                        v
        exact-ID resolution + entry / relation materialization
                        v
       searchable entry list + loose-tag navigation + retry queue
```

The stages should have narrow responsibilities:

| Stage | Input | Output | Must not do |
| --- | --- | --- | --- |
| `enumerate-papers` | source cursor, snapshot manifest, or explicit seeds | durable paper identities and checkpoints | decide a model name or download all linked files |
| `ingest-paper` | one exact paper identity | artifact evidence and discovered direct links | assume every paper describes exactly one entry |
| `enrich-resource` | one admitted public resource URL | a bounded artifact observation and outgoing direct links | recurse indefinitely or access private content |
| `materialize-entry` | artifacts, exact IDs, relation evidence | entry-facing resource graph | merge entries by fuzzy name equality |
| `tag-entry` | entry ID and editorial tags | loose navigation metadata | filter ingestion or claim scientific truth |

`ingest-paper --input PAPER.json` now implements the source-record composition for one
normalized exact paper observation. It stores the immutable paper evidence under its
original source identity, records every direct source/text URL with a relation and
locator, conservatively derives a candidate only if the source supplied none, and emits
a portable seed plus a no-side-effect action plan. The command's `--dry-run` validates
that path without opening a registry; the normal form persists evidence and bounded
frontier work but does not materialize an entry bundle. Source adapters and `SyncEngine`
remain the checkpointed enumeration layer, the URL frontier remains bounded enrichment,
and the exporter provides exact cross-artifact resource evidence without a global graph.
`build-entry-corpus` remains a separate explicit publication action.

### Portable entry-seed contract

The boundary between a paper resolver and the entry builder should be a small,
serializable seed. This lets a human review one paper's proposed actions, lets an
automated worker resume an exact work item, and lets the future enumerator emit the same
format at scale. The in-progress entry builder uses the following shape:

```json
{
  "source": "arxiv",
  "source_record_id": "1706.03762v7",
  "canonical_url": "https://arxiv.org/abs/1706.03762",
  "title": "Paper title",
  "kind": "paper",
  "identifiers": [{"namespace": "arxiv", "value": "1706.03762"}],
  "tags": ["bioML", "protein design"],
  "links": [
    {
      "url": "https://example.org/project",
      "relation": "project_page",
      "locator": "abstract:project URL",
      "crawl": true
    },
    {
      "url": "https://example.org/weights.safetensors",
      "relation": "weights",
      "locator": "README:weights",
      "crawl": false
    }
  ],
  "models": [
    {
      "local_id": "method-a",
      "name": "Method A",
      "aliases": ["M-A"],
      "identifiers": []
    }
  ]
}
```

`source` plus `source_record_id` gives the observation its durable provenance;
`models` is the list of candidate entry subjects. Each model's identifiers name that
subject, not the supporting paper: omit them when no exact subject identifier exists.
Every `links` row carries a relation, locator, and crawl policy. An optional
`model_local_ids` array scopes a resource to the exact declared subjects in the same seed;
when omitted, the link is record-wide. This prevents a multi-model catalog from attaching one
model's checkpoint to every sibling, while preserving a page-level paper or license that the
source explicitly declares for the whole collection. The seed must never invent an exact
identifier or mark a third-party link as official merely to make the bundle look complete.

The current builder treats an absent or empty `models` list as no entry candidate.
`ingest-paper` makes that choice explicit: it stores the paper and its resource evidence
without materializing an entry when neither source declaration nor the conservative
extractor supports a subject. For an admitted subject, the planner turns the seed into
`upsert_entry`, `attach_resource`, and bounded `resolve_resource` work; a non-fetchable
weights reference gets attached but never scheduled for a byte download.

### Required script properties

Every stage must be:

- **idempotent:** rerunning the same seed or page must not create duplicate records;
- **restartable:** a durable checkpoint or work item records progress after each safe
  commit;
- **bounded:** page count, link depth, response size, and retry count are explicit;
- **evidence-preserving:** raw/normalized observation, source, locator, and timestamp
  survive any derived entry view;
- **exact-ID-first:** aliases aid search but do not auto-merge entities;
- **failure-tolerant:** one malformed record becomes retryable/quarantined work rather
  than invalidating unrelated ingestions; and
- **rights-aware:** private content, credentials, robots rules, rate limits, licenses,
  and export policy are enforced at the artifact boundary.

These properties matter more now than raw throughput. A script that correctly ingests
ten representative papers is a better foundation than a large one-off download that
cannot be resumed, audited, or reused for a new source.

## Path to complete paper ingestion

“Ingest every paper” is a future coverage objective, not an instruction to start a
download immediately. Reaching it has two independent dimensions:

1. **Enumeration coverage:** traverse the declared inventory of each chosen paper
   source—API pages, OAI-PMH records, snapshots, or releases—with checkpoints and
   reconciliation. This creates a durable worklist of paper identities.
2. **Per-paper entry processing:** feed each identity through `ingest-paper`, then
   bounded resource enrichment and entry materialization.

Keeping these separate avoids an all-or-nothing migration. We can validate extraction,
identity resolution, relation semantics, and storage on individual papers today. Later,
an enumerator can enqueue millions of exact paper IDs without changing what “ingest one
paper” means.

The recommended rollout is:

1. Establish a small, diverse seed set of papers and execute the full loop manually or
   through an explicit seed file. Use it to validate the minimum entry shape and link
   relation vocabulary.
2. Add an auditable work queue with states such as `discovered`, `ingesting`,
   `enriched`, `materialized`, `retryable_failure`, `blocked_by_policy`, and `complete`.
   State is about the work item, not a judgment that a paper is globally complete.
3. Extend the `ingest-paper` fixture suite through papers with no links, multiple
   entries, versions, official and unofficial code, a model card, private/dead links,
   and conflicting identities.
4. Attach one broad source enumerator at a time. It only creates exact paper work items
   and records source coverage; it does not need to retrieve every full text or linked
   binary.
5. Add capacity controls and source-specific reconciliation before calling an
   enumeration complete. “Complete” always means complete for a stated source release,
   date window, and script version—not all research ever published.

Bulk source snapshots, full-text corpora, provider feeds, and archive-wide projections
remain valuable future tools. Introduce them only when a measured coverage or throughput
need justifies their operational and rights cost.

## Non-goals for the current phase

- Building or downloading a complete PDF/full-text corpus.
- Downloading repository histories, model-weight files, datasets, or arbitrary
  supplementary files.
- Crawling the open web beyond bounded, evidence-linked resources.
- Creating a universal controlled vocabulary for disciplines or techniques.
- Treating every paper as a model entry.
- Making global-completeness claims from partial source scans.
- Backward-compatibility layers for an early schema when a simpler forward design is
  clearly better.

## Decisions to keep explicit as the system grows

Several choices will become important soon. Keep them as explicit, versioned policy
rather than burying them in an extractor:

- which link predicates are strong enough to surface as “official”;
- what evidence level is required to create, merge, split, or retire an entry;
- which source-specific identifiers are exact and which are mutable labels;
- whether an entry can represent a technique, architecture, trained release, or a
  broader family, and how the visible UI distinguishes them;
- which fields may be exported publicly under each source's rights terms; and
- what “paper ingestion complete” means when a source, link, or text is inaccessible.

The answer to these questions can evolve. The invariant is that the user should be able
to inspect why a resource appears in an entry, and the future corpus-scale runner should
be able to reuse the same small-paper workflow unchanged.
