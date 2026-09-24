# Source strategy

This document is a capability and future coverage inventory. The current product
workflow is [entry-first paper ingestion](entry-first.md): one exact paper seed,
its directly supported resources, and a useful entry. A source being enabled in
`config/sources.toml` means the adapter can be invoked; it does **not** authorize or
require a global crawl, full-text download, or bulk transfer during the present phase.

MODELOME's target source graph combines model catalogs, scholarly indexes, framework
registries, code archives, domain corpora, and provider APIs. No one source is canonical
for all neural-model entities. The current default configuration has **356 enabled
source entries** (341 loadable without provider credentials; a count of `config/sources.toml`,
not a claim that upstream inventories have been exhausted) and enables Hugging Face,
Kaggle Models, CivitAI, OpenCSG Hub, a bounded ModelScope catalog plane, NVIDIA NGC's
guest-visible current `MODEL` catalog, first-party NVIDIA NeMo checkpoint tables, Ollama's library cards, Cloudflare Workers AI's
current catalog, Zenodo's Model-resource catalog, and OpenRouter's public current model catalog; and,
when `REPLICATE_API_TOKEN` is explicitly configured, Replicate's displayworthy public catalog; when
`OPENAI_API_KEY` is explicitly configured, OpenAI rows limited to its public-owner policy; or when
`ANTHROPIC_API_KEY` is explicitly configured, Anthropic's documented provider-available model list;
or when `GEMINI_API_KEY` is explicitly configured, the Google Gemini Models API list;
or when `GROQ_API_KEY` is explicitly configured, Groq's documented all-active-models list;
Mistral's public model documentation, Cohere's public live/deprecated model overview, and
OpenAI's public all-models documentation (including its documented deprecated cards), and
Amazon Bedrock's public model-card catalog plus SageMaker JumpStart's public dated model table;
OpenAlex publication and update feeds; arXiv, both public OpenReview APIs, Crossref,
Europe PMC, DataCite, and PMC;
bioRxiv and medRxiv metadata and publication-link streams; Semantic Scholar and PubMed
bulk controls; global OSF community-preprint metadata; first-party EarthArXiv and HAL OAI-PMH
streams and the PLOS publisher API; Common Crawl WET controls; GitHub's public-repository API and
GH Archive public-event controls; and the Software Heritage full-graph origin export. Linked
GitHub/public pages can be enriched through the URL frontier. Epoch AI is configured separately as
a read-only benchmark. The generic provider JSON adapter remains available for separately
documented sources; OpenRouter uses a dedicated adapter because its catalog declares exact
Hugging Face artifact IDs and router aliases, while Replicate's token-gated adapter retains its
direct code/paper/weight/latest-version fields; the OpenAI adapter excludes non-public owner rows
before persistence; Anthropic's header/version/cursor contract and Gemini's query-key/page-token
contract, plus Groq's header-authenticated active-model list, are expressed declaratively.
Hugging Face's own public model listing preserves
direct card/config paper and repository URLs as typed references when their canonical IDs prove
the relation. Each section below states which mechanics exist
and which remain roadmap work.

Additional enabled inventories cover DeepChem Mol2Vec, Tencent GROVER, Microsoft
VQ-Diffusion, AlphaChip, PaddleNLP sentiment, sherpa source separation, and OpenVLA.
They also cover ADMET-AI Chemprop ensembles, DIPY neuroimaging weights, PaddleNLP
knowledge-mining checkpoints, Octo policies, and sherpa audio tagging.
The GitLab adapter scans public project release metadata one request per page and emits
low-confidence model-file candidates. The Cohere and Mistral Models API proposals
remain opt-in because their authenticated lists can include caller-owned models.

When an adapter is eventually used for corpus enumeration, it must traverse its complete
configured upstream scope without a model-name, method-keyword, subject, or journal
allowlist and retain the upstream ID, source revision or observation time, retrieval
checkpoint, and supplied rights declarations. Where a provider enforces a documented
bounded window rather than exposing a full traversal, configuration and reporting must say
so explicitly, as ModelScope does. That later enumerator only produces exact paper/artifact
work items; the same bounded per-paper ingestion stage handles entry
materialization. Scope classification happens after acquisition. Deep-learning and
neural-network architectures, trained instances, checkpoints, and hybrids with a
substantive neural component can become central model entries. A first-party framework
catalog can also declare a classical estimator, transformer, or composition component as
a documented ML-technique entry, with an explicit source tag/status rather than a
fabricated trained-release claim. Scientific uses of “model” and “diffusion” do not
qualify merely from those words. The future entry layer can still link a broader research
object when it is useful; it must not misrepresent it as a neural-model claim. The current
schema keeps license data inside raw evidence; normalized rights projections remain future
work.

The links below point to primary documentation or source repositories. Rate limits and
commercial terms change; adapters must treat the linked policy as authoritative, honor
response headers, and keep limits configurable rather than embedding them in logic.

## External benchmark (never ingested)

### Epoch AI Models

- **Purpose:** use Epoch's curated public model corpus as an independent minimum-recall
  benchmark for important models, not as a discovery feed or canonical registry.
  “All models” still means all rows in that evaluated Epoch corpus, not all models that
  exist.
- **Execution:** `modelome benchmarks` lists the benchmark configuration. The command
  `modelome benchmark --name epoch --minimum-recall 1.0` fetches the current corpus,
  evaluates the existing store, and returns nonzero when recall is below the selected
  threshold.
- **Anti-contamination:** benchmark rows are held only for the evaluation and are never
  written to Parquet as artifacts, revisions, models, aliases, identifiers, releases,
  links, or provenance. Ordinary `modelome sync` never requests Epoch. Freshly initialized
  stores therefore contain no Epoch evidence, and benchmark matches require evidence
  discovered by an ingestion source.
- **Reporting:** record the benchmark retrieval time, corpus size, misses, threshold,
  and achieved recall. Attribute Epoch's [AI Models dataset](https://epoch.ai/data/ai-models)
  and follow its current terms when publishing benchmark results.
- **Independence:** benchmark names and metadata must not become queries, extractor
  vocabulary, source configuration, crawl seeds, or identity hints. A miss should drive
  investigation and broader source/adaptor work, not a one-off hardcoded model entry.

## Available continuous-source adapters

These sources are present and enabled in `config/sources.toml`. They are candidates for
the later enumeration layer; do not schedule all of them while validating the
single-paper entry loop.

### Hugging Face Hub

- **Enumeration:** page through the official [Hub API](https://huggingface.co/docs/hub/api)
  model listing with `full` and `cardData` requested. Revisit records by last-modified
  overlap and retain links to card/config files listed by the API.
- **Coverage:** public model repositories, API-returned card/config metadata, library and
  pipeline tags, paper links, revision-pinned sibling weight URLs, and `base_model`
  lineage across modalities. A listed URL is not proof that MODELOME downloaded or
  verified its bytes.
- **Identity:** model repository ID plus immutable commit SHA when the API supplies one.
  A repository, revision, and model release are related identities, not synonyms.
- **Historical files:** an operator can enable `include_revisions` and
  `include_revision_files` to list exact weight paths at commit SHAs without fetching
  blobs. `max_revision_tree_pages` bounds each tree; capped or unavailable trees
  retain observed paths with `weight_files_complete=false`.
- **Rights:** metadata and files follow the repository's declared license and the
  [Hub terms](https://huggingface.co/terms-of-service); public visibility is not a blanket
  reuse license. Preserve unknown, custom, gated, and per-file license states.
- **Access:** the default source is deliberately unauthenticated so a personal token
  cannot expose private repositories to a public registry. An operator may configure
  `auth_env` explicitly; private results are still discarded unless `include_private`
  is also explicitly true for a private deployment. Paginate, respect tiered limits and
  429 headers, and do not crawl rendered model pages.

### Papers With Code official paper-to-code archive

- **Enumeration:** obtain the public Hugging Face dataset metadata, pin its immutable
  revision, verify the declared Parquet path, and import its complete paper-to-code
  snapshot. No paper title, method name, or repository query is used to select rows.
- **Coverage:** one paper artifact and one concrete code-repository artifact per distinct
  item in the source-declared `is_official=true` relation set. The edge is retained as
  `official_implementation` evidence and is materialized without crawling the linked
  paper or repository.
- **Identity:** PWC paper URL and repository URL are source identities; supplied arXiv
  IDs are normalized as cross-source artifact identifiers. A PWC edge does not assert
  that a paper's title or an arbitrary method label is a model identity.
- **Quality and rights:** the archive is [CC-BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
  The separate PWC methods archive is never blindly ingested because its current public
  snapshot contains plainly unrelated, vandalized entries. Its enabled documented and
  candidate tiers therefore apply explicit, reproducible structural and anti-spam rules.

### PWC methods with arXiv verification

- **Enumeration:** the raw PWC methods archive is never accepted directly. A separate
  weekly source retains only method records with a valid method URL, no phone-number
  payload, at least two meaningful tokens shared with their declared source-paper title,
  and an arXiv API title that substantially agrees with that declared source title.
- **Coverage:** admitted rows are documented method identities with both their PWC method
  URL and canonical arXiv ID. They link to the PWC paper and verified arXiv source paper,
  but do not assert a release, weights, or an official implementation.
- **Quality and rights:** every rejected row is counted in the snapshot checkpoint, and
  the deterministic validation policy is part of the source signature. This is a
  conservative salvage tier, not a rehabilitation of the contaminated raw dump; retain
  PWC's CC-BY-SA attribution and share-alike obligations in exports.

One adjacent candidate tier retains structurally valid, phone-free, arXiv-linked method
records without title verification. A second, lower-confidence tier retains the same
paper-linked shape even when its declared source paper is not arXiv-hosted. Both preserve
their exact PWC method, PWC-paper, and source-paper links, but remain `candidate`
identities and must not be presented as documented or released models.

### PWC evaluation-table candidates

- **Enumeration:** the frozen evaluation-table dataset is enumerated shard by shard from
  its public Hugging Face metadata at an immutable revision. Every table row is considered
  structurally, without a task, domain, paper, or model-name filter.
- **Coverage:** a retained candidate must have a bounded, phone-free label plus a valid
  linked paper URL and paper title. Repeated observations of the same label and paper are
  collapsed while preserving task context and any reported-code URLs. Reported code is
  not an `official_implementation` assertion.
- **Quality and rights:** table content can include aliases, ablations, or malformed
  entries, so these are `candidate` identities only, with no release or official-code
  claim. Preserve the source revision and PWC's CC-BY-SA attribution/share-alike terms
  in every derivative export.

### Wikidata machine-learning-model class

- **Enumeration:** a bounded public SPARQL query retrieves English-labelled items that
  Wikidata classifies as instances or subclasses of its `machine learning model` class.
  QIDs are source identities, and the query contains no model-name list.
- **Coverage:** this is a compact, independent historical identity plane. Its records
  are `documented`, not asserted as released weights or provider-supported artifacts.
- **Quality and rights:** the narrower class is the higher-precision tier; the broader
  neural-network hierarchy is separately labeled as lower-certainty documented evidence
  because it includes visibly misclassified entries. Wikidata's structured data is CC0;
  individual linked works and artifacts keep their own terms.

### Wikimedia neural-network architectures

- **Enumeration:** a public MediaWiki category-members query pages every main-namespace
  member of the generic neural-network-architectures category. The category name and
  pagination cursor are the source definition; it contains no model-name seed list.
- **Coverage:** entries are a free, independent historical evidence plane for named
  architectures and families. They are `documented`, never provider release or weight
  assertions, and may include generic concepts alongside individual models.
- **Quality and rights:** page IDs are source identities and page titles are labels.
  Wikimedia content has its own reuse terms; this registry retains provenance and links,
  not an implied blanket license for linked material.

The broader neural-network hierarchy is also retained as a distinct `documented` tier for
historical architecture recall. It has the same QID identity scheme, but its community
classification is never elevated to a provider release claim and should be corroborated by
papers, code, model cards, or weights as those sources arrive.

### OpenAlex

- **Enumeration:** use cursor pagination from the [Works API](https://help.openalex.org/api/)
  for daily publication windows and the implemented inclusive historical-window command.
  The official [snapshot](https://help.openalex.org/access/snapshot/) is the preferred
  future route for a full backfill at scale; it is not loaded by the current adapter.
  An empty model-name filter is intentional, so extraction sees the entire configured
  window rather than a name-selected subset.
- **Coverage:** cross-disciplinary papers, preprints, citations, concepts/topics,
  authors, institutions, locations, and external IDs. It provides the broad paper layer
  needed to find models whose authors never publish a model card.
- **Identity:** OpenAlex work ID plus DOI, PMID, PMCID, arXiv ID, and other supplied
  external identifiers. Preserve merged-location and source relationships.
- **Rights:** OpenAlex metadata is [CC0](https://github.com/ourresearch/openalex-docs/blob/main/license.md).
  Linked or hosted [full text retains its original license](https://help.openalex.org/access/fulltext/).
- **Access:** use an API key for production quotas and keep page budgets configurable.
  `MODELOME_CONTACT_EMAIL` can supply operator contact metadata where the API accepts it; it
  is not a substitute for the current API-key quota model. Prefer the bulk snapshot for
  historical scale.

### bioRxiv and medRxiv metadata and publication links

The default catalog enables four first-party streams: one unfiltered `details` stream
and one `pubs` stream for each server. They retain every API-returned preprint version,
the JATS full-text link, and DOI-based journal-publication relations. Each stream uses
its own frozen inclusive UTC window, cursor offset, overlap, high-water mark, and
pagination validation. The complete operational and identity contract, including the
remaining TDM and reconciliation work, is specified in the Priority 0 section below.

### OSF community preprints

The default `osf-preprints` stream calls OSF's top-level public preprint endpoint with
no provider, subject, or model-term filter. It therefore covers every OSF-hosted
community, including the active rXiv family and historical records from communities that
have since moved to another platform. It freezes an inclusive UTC `date_modified` window
across numbered pages, preserves each OSF preprint ID, provider, preprint DOI, metadata
DOI, description, tags, and linked OSF project, and replays a window from page one if its
declared total changes. This source is metadata-only; a preprint HTML page or linked
project is evidence to enrich later, not proof that its contents are freely reusable.

### EarthArXiv

EarthArXiv's current California Digital Library Janeway repository publishes a supported
[OAI-PMH feed](https://eartharxiv.org/api/oai/). The default stream uses its unfiltered
Dublin Core records and resumption tokens, preserving OAI and object IDs, DOIs, full-text
links, creators, subjects, rights, abstract, and datestamps. This supplements the legacy
EarthArXiv records still visible through OSF and avoids scraping its search pages.

### HAL

HAL's first-party [OAI-PMH server](https://api.hal.science/docs/oai) exposes every public
research output across its archive and portals. The enabled adapter begins at HAL's
published `2002-09-23` lower boundary, uses date-granularity `oai_dc` requests, freezes
each complete window while following opaque resumption tokens, and applies no subject,
institution, collection, venue, author, type, or model-term selection. It preserves the
HAL OAI ID, version and base document IDs, source sets, creators, abstract, dates, DOI,
rights, and HAL/full-text links. HAL declares no OAI deletion records, so the adapter
never invents a removal from absence; it only retains a tombstone if the protocol itself
provides an explicit deleted header.

### PLOS

PLOS's own [Search API](https://api.plos.org/solr/faq) exposes its complete publisher corpus.
The enabled source requests every `doc_type:full` record in publication-date/DOI order with no
journal, topic, author, or model-term selector. It freezes a closed UTC publication window,
keeps the reported result total, and restarts at offset zero if that total changes before the
window completes. It preserves publisher DOI, title, authors, abstract, article type, journal,
subjects, ISSNs, copyright statement, and non-crawling DOI/article/JATS links. The few structural
`Issue Image` records are retained as `other` artifacts rather than mislabelled as papers. PLOS
publishes full text separately through its stated TDM route; this source records metadata and
links, and neither bulk-downloads nor scrapes article pages.

## Priority 0: global scholarly and biomedical publication plane

This plane is the first coverage gate, not optional enrichment behind OpenAlex. Its
continuous APIs, bulk-control adapters, bounded bulk loaders, and arXiv/PMC historical
bootstraps are implemented and enabled. Work still remains to finish corpus-scale
searchable projections, reconcile every source's complete inventory, and ingest bulk
preprint TDM packages. Every source must ingest its full corpus without an AI,
bioinformatics, subject-category, publication-type, or journal query. Neural-model scope
is decided from retained metadata and text after ingestion.

### Semantic Scholar Academic Graph datasets

Semantic Scholar is a first-class cross-disciplinary enumerator, not merely a search or
recommendation service:

- **Bootstrap (control and landing implemented):** pin one release from the official [Datasets
  API](https://api.semanticscholar.org/api-docs/datasets) and exhaust every shard of
  `papers`, `abstracts`, and `paper-ids`. Retain the release ID and per-dataset manifests,
  verify every downloaded object, and account for every parsed or quarantined record.
  Use `corpusId` as the dataset paper key, join `paper-ids` for SHA-based aliases, and
  preserve DOI, arXiv, PubMed, and other external IDs supplied by `papers` and
  `abstracts`. Do not select shards or records by field or keyword.
- **Updates (control, transfer, and stateful projection implemented):** the daily MODELOME job polls the available release list. When a newer release
  appears, request the official incremental diffs from the committed release to that
  release for all three required datasets. Apply each sequential update file as
  insert-or-replace by primary key and each delete file as explicit deletion evidence.
  Publish the new Semantic Scholar release watermark only after every required shard and
  diff is verified and committed; a partial dataset must remain resumable and invisible
  as a complete release.
- **Model candidates (implemented):** every completed normalized projection is scanned in
  bounded batches by the versioned, name-agnostic introduction-cue extractor. Candidate
  assertions are published as a separately sealed Parquet artifact with exact spans,
  supporting sentences, upstream identifiers and URLs, tombstone counts, and the source
  projection identity. Candidate spellings are not automatic cross-paper merge keys.
- **On-demand enrichment:** use the Academic Graph paper or batch endpoint only after
  MODELOME already has an exact Semantic Scholar or external paper identifier. Keyword search,
  bulk search, recommendations, and citation expansion are not enumeration mechanisms
  and must not become discovery seeds.
- **Targeted citation edges (opt-in):** a custom source with adapter
  `semantic_scholar_citation_graph`, an exact `paper_id`, `paper_url`, and `paper_title`
  can page that paper's `citations` or `references` endpoint. It preserves exact
  paper IDs and citation direction, with a bounded offset checkpoint; it does not
  expand the default global catalog or infer model identity from paper titles.
- **Bulk citation edges (opt-in):** the Semantic Scholar release also exposes
  `citations` shards. Bulk intake can preserve their exact citing and cited paper
  IDs and raw edge payload, but citation edges are not yet published into the
  paper projection; the large dataset stays outside the default download set.
- **Rights:** require an API key for full dataset downloads and diffs. Preserve the
  README and license delivered with each dataset release; the core paper, abstract, and
  paper-ID datasets currently declare ODC-By attribution terms, while API access is also
  governed by Semantic Scholar's [API license agreement](https://api.semanticscholar.org/license/).
  Review those terms for the intended public or commercial deployment and retain required
  attribution rather than assuming all fields are unrestricted.

### bioRxiv and medRxiv

bioRxiv and medRxiv are separate first-class source namespaces with the same operational
contract:

- **Enumeration (implemented):** scan the official [`details` API](https://api.biorxiv.org/) for
  each server with inclusive UTC calendar-date intervals and cursor pagination. The
  response reports current-page `count` (at most 30) and window `total`. Do not send its
  optional category filter. Keep independent checkpoints for bioRxiv and medRxiv.
- **Identity and versions (implemented):** use `(server, normalized DOI)` for the
  manuscript identity and `(server, normalized DOI, integer version)` for the immutable
  preprint-version artifact. Retain every version, its posting date, type, license,
  abstract, category, funding, and API-supplied JATS path; a later version must never
  overwrite an earlier one.
- **Full text (targeted follow-up implemented):** retain each record's JATS path as a typed
  full-text link. An optional, separately checkpointed adapter follows first-party XML
  paths from details records and keeps only explicit model-related supplementary links.
  It is disabled in the default catalog because it needs one rate-limited XML request
  per eligible paper. Bulk ingestion from the official
  [bioRxiv](https://www.biorxiv.org/tdm) and
  [medRxiv](https://www.medrxiv.org/tdm) requester-pays S3 TDM collections remain roadmap
  work. Those collections publish monthly processed `.meca` packages containing
  full-text XML and PDF plus supplied images and supplements. Record TDM-package identity
  and content digest separately from the API record. TDM access permits machine analysis;
  public outputs must link to the hosted preprint and obey the version's author-selected
  license rather than republish source text.
- **Publication links (implemented):** page the separate `pubs/{server}` API, which
  returns at most 100 preprint-to-journal mappings per call, under its own checkpoint.
  Accept the first-party preprint DOI field, and store the published DOI, journal, and
  both dates as a source-backed `published_as` relation. The journal DOI is the relation
  target, not another identifier for the preprint artifact: a publication link does not
  merge or replace the preprint version with the journal article.
- **Daily update (implemented):** for both `details` and `pubs`, end at yesterday's
  closed UTC date and freeze the inclusive bounds across pagination and retries. After
  the initial configured lookback, start at
  `prior_watermark + 1 day - overlap_days`. The enabled metadata
  streams replay two closed dates; the publication-link streams replay 90 days to catch
  links registered well after a preprint appears. Advance the offset by the actual
  collection length, validate status/cursor/count/total, and restart the frozen interval
  at cursor zero if `total` changes mid-scan. Advance that endpoint's high-water mark only
  after every expected record is accounted for. Replay is idempotent on the version tuple
  or DOI pair plus normalized content hash.
- **Reconciliation (planned):** periodically replay closed calendar-month intervals and compare
  their API counts with locally retained version tuples, and cycle through the full
  `pubs` history because a journal link can be learned after its published-date window.
  Reconcile full-text availability against the later monthly TDM inventory. Absence from
  an overlapping date query or a not-yet-published TDM package is not deletion evidence.
  Preserve withdrawals as status changes and tombstone or suppress content only on
  explicit authoritative removal evidence; retain the immutable observation and reason
  when policy permits.

### OSF-hosted rXiv communities

OSF's public API exposes one complete preprint collection whose records identify their
community provider. The implemented `osf-preprints` source deliberately scans that
collection rather than configuring a brittle list of individual communities such as
PsyArXiv, SocArXiv, MetaArXiv, or historical EarthArXiv records.

- **Enumeration (implemented):** page the global
  [`/v2/preprints/` API](https://api.osf.io/v2/preprints/) in ascending
  `date_modified` order with closed UTC timestamp boundaries. Keep no provider or
  discipline filter. The API's next-page link and declared total are validated; a total
  change restarts the immutable window before checkpoint advancement.
- **Identity and evidence (implemented):** retain the OSF preprint ID, provider ID,
  canonical preprint URL, OSF-generated preprint DOI, metadata DOI, description, tags,
  timestamps, and related OSF project URL. A DOI is an identifier for that preprint
  record; it does not establish a model identity or merge a project with a paper.
- **Scope and rights:** OSF hosts communities beyond the rXiv family, so neural-model
scope is decided after acquisition. Public metadata or a reachable project does not
grant a blanket right to redistribute manuscript text, files, code, or weights.

### EarthArXiv Janeway repository

EarthArXiv now has its own first-party Janeway instance, with active scientific ML and
GeoAI preprints outside the historical OSF collection.

- **Enumeration (implemented):** query the repository's unfiltered `oai_dc`
  [`OAI-PMH feed`](https://eartharxiv.org/api/oai/) over frozen closed UTC windows;
  then follow its opaque resumption tokens until exhaustion. OAI's `noRecordsMatch`
  response completes an empty window without treating it as an error.
- **Identity and evidence (implemented):** retain the OAI identifier, EarthArXiv object
  ID, every declared DOI, canonical object page, full-text download link, authors,
  subjects, rights, abstract, posting date, and OAI datestamp. Explicit OAI deletions
  remain deletion evidence rather than inferred absence.

### PubMed, PMC, and Europe PMC

- **PubMed/MEDLINE (implemented):** ingest NLM's [complete annual XML baseline and daily update
  files](https://pubmed.ncbi.nlm.nih.gov/help/#download-pubmed-data), without a search
  query. Daily files contain new, revised, and deleted citations, so apply them in file
  sequence, retain the PMID as source identity, and process explicit deletions rather
  than inferring them from absence. Preserve DOI, PMCID, publication status, MeSH,
  databank, and secondary-source links as cross-source evidence.
- **PMC (OAI-PMH JATS implemented):** use the approved PMC OAI-PMH service with the
  `pmc` metadata format and the `pmc-open` rights set, which is not a topic filter. The
  daily source freezes closed UTC windows, handles resumption tokens and deletions, and
  retains bounded full JATS text, identifiers, authors, journal metadata, license, and
  implementation links. A separate bootstrap begins at the repository's declared
  earliest datestamp and resumes toward the last closed day. PMC Article Datasets on
  AWS inventory reconciliation remains additional redundancy rather than a prerequisite
  for daily JATS ingestion.
- **Europe PMC (REST daily and historical bootstrap implemented):** use the [Articles REST API](https://europepmc.org/RestfulWebService),
  OAI-PMH, and [bulk downloads](https://europepmc.org/downloads) as an independent
  biomedical metadata, full-text, preprint, citation, and data-link layer. Cursor through
  unfiltered `UPDATE_DATE` windows with a committed multi-day overlap. A separate
  resumable bootstrap scans closed calendar months from a configurable historical floor,
  and `bootstrap --source europe-pmc` runs it with a finite page budget. Use the
  weekly open-access XML and full-text metadata files for reconciliation. Preserve PMID,
  PMCID, DOI, Europe PMC/PPR identity, preprint versions, and links to journal-published
  versions. Europe PMC corroborates and enriches direct bioRxiv/medRxiv observations; it
  does not replace either first-party enumerator.

## Priority 1: framework and model registries

The default catalog now structurally enumerates the official scikit-learn, sktime, aeon, Darts,
PyOD, PyGOD, Statsmodels, tslearn, AutoGluon Tabular, PyTorch Forecasting, GluonTS, NeuralForecast, StatsForecast, PyTorch Tabular, TorchGeo, and Segmentation Models PyTorch APIs, Torchvision family pages and static weight enums, Torchaudio, Keras Applications, KerasHub family pages and static source-declared presets, PyTorch Geometric, TorchDrug,
DeepChem, fastText, Transformers, PaddleNLP transformer families and pretrained embeddings,
Sentence Transformers, spaCy pipelines, Stanza historical resources, and Diffusers index documents.
Extraction rules describe document structure and contain no model names; selector drift
fails closed instead of publishing a false empty snapshot. Every matched occurrence and
the complete bounded source document are retained. Package-version introspection,
weight-enum expansion, and component/checkpoint resolution remain follow-on work.

Other recently enabled source entries add these independently bounded planes:
[`scenic-baseline-model-zoo`](https://github.com/google-research/scenic/blob/main/scenic/projects/baselines/README.md)
reads Scenic's source-declared checkpoint tables;
[`stardist-pretrained-models`](https://github.com/stardist/stardist/blob/main/stardist/models/__init__.py)
reads literal 2D/3D release mappings;
[`coqui-tts-model-registry`](https://github.com/coqui-ai/TTS/blob/dev/TTS/.models.json)
reads Coqui's model metadata; [`dgllife-pretrained-graph-checkpoints`](https://github.com/awslabs/dgl-lifesci/blob/master/python/dgllife/model/pretrain/property_prediction.py)
reads literal graph-property checkpoint mappings; [`unimof-checkpoints`](https://github.com/dptech-corp/Uni-MOF/blob/main/README.md)
reads release assets; [`openfold-documented-checkpoints`](https://github.com/aqlaboratory/openfold)
reads documented model checkpoints; and `argus-robotics-checkpoints` retains the
configured Argus checkpoint inventory. `kaggle-models` retains every public variation
and exposed historical version. `acl-anthology-all-collections` freezes ACL Anthology's
official Git tree XML manifest, visits one collection per page (up to the configured
20,000-file bound), and skips unchanged collection files by digest.
[`oci-generative-ai-pretrained-models`](https://docs.oracle.com/en-us/iaas/Content/generative-ai/pretrained-models.htm)
and [`azure-foundry-models-sold-by-azure`](https://learn.microsoft.com/en-us/azure/foundry/foundry-models/concepts/models-sold-directly-by-azure?view=azureml-api-2)
add hosted-provider catalog evidence; and [`river-ml-technique-overview`](https://riverml.xyz/latest/api/overview/)
adds documented anomaly-detection and clustering components as techniques, without
treating those components as trained releases.

Nine additional source IDs extend the framework and checkpoint plane:
`cellpose-website-checkpoints` retains the legacy Cellpose website weights declared in
its v3 and restoration guides (not the separately hosted v4 built-ins);
`pytorch-geometric-gpse-checkpoints` parses literal `GPSE.url_dict` checkpoint URLs;
`nnunet-v1-pretrained-models` records the authors' task bundles in Zenodo record 3734294
(not nnU-Net v2); `compvis-latent-diffusion-downloads` reads direct bundle URLs from
CompVis's pinned download script; `diffusion-policy-checkpoints` records scored epoch
checkpoints from the project's indexed directories; `chemprop-chemeleon-checkpoint`
retains the documented CheMeleon weight file; `mindspore-modelzoo` follows only direct
checkpoint files below MindSpore's official model-zoo tree; and
`google-robotics-transformer-checkpoints` enumerates RT-1 SavedModel directories that
contain both the policy specification and variables index. `torch-hub-listing-extra`
adds literal repository/entrypoint declarations from PyTorch Hub's curated listing and exact ref-pinned `torch.hub.load` code calls,
excluding TorchVision to avoid duplicating its dedicated weight registry.

Six more source IDs add specific checkpoint and hosting evidence:
[`alphafold-parameter-archive`](https://github.com/google-deepmind/alphafold/blob/main/scripts/download_alphafold_params.sh)
retains the one parameter-archive URL literally selected by AlphaFold's pinned script,
without inspecting the archive; [`graphgps-ogblsc-pretrained-asset`](https://github.com/rampasek/GraphGPS/blob/main/README.md)
records only the documented GPS-deep OGB-LSC inference archive;
[`gensim-downloader-models`](https://github.com/RaRe-Technologies/gensim-data/blob/master/list.json)
retains complete single-file gzip models from the manifest's `models` section, including
declared MD5s, while excluding corpora and multipart entries;
[`aws-bedrock-region-matrix`](https://docs.aws.amazon.com/bedrock/latest/userguide/models-region-compatibility.html)
attaches published Region and lifecycle availability to exact Bedrock model-card IDs;
[`opencv-dnn-model-index`](https://github.com/opencv/opencv/blob/4.x/samples/dnn/models.yml)
records aliases and their direct declared DNN download URLs (which may be hosted by
third parties), without fetching weights; and
[`google-mediapipe-model-catalog`](https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/models.md)
retains exact `.tflite` and `.task` links from the legacy Solutions model list. That page
documents support that ended in 2023; it is not a complete index of newer MediaPipe tasks.

The enabled hosted-model additions are `together-models`, `cerebras-models`,
`deepseek-models`, and `sambanova-cloud-models`. Each API requires its provider key and
records exact current model IDs as provider-availability evidence, not as origin-model,
paper, or checkpoint claims.
`cerebras-public-model-documentation` separately retains model IDs from Cerebras' public
model overview, while `perplexity-hosted-model-documentation` retains only the
`perplexity/...` entries in Perplexity's public router-model table. These public pages
document hosted offerings and do not establish downloadable weights.

| Source | Machine enumeration | Stable evidence and coverage | Rights/access notes |
| --- | --- | --- | --- |
| [scikit-learn API](https://scikit-learn.org/stable/api/index.html) and retained [versioned documentation](https://scikit-learn.org/0.18/modules/classes.html) | **First-party public ML-component indexes enabled.** Every structurally listed class page in the current API and the retained 0.18, 0.20, 0.22, 0.24, 1.0, 1.2, 1.4, 1.6, and 1.8 snapshots across learning, estimator, neural-network, decomposition, clustering, transformation, and composition modules becomes a `scikit-learn:machine-learning-component` identity derived from its full API path. | Adds current and decade-scale historical evidence for classical estimators, neural-network classes, unsupervised methods, feature/model transformations, and compositional techniques without a model-name list or package execution. The identical qualified path is an exact cross-version bridge; version-specific class pages remain independently retained evidence. | The selected snapshots preserve anchor points, not every patch release or a source-wide proof of historical completeness. A documented component is not asserted to be a trained model, checkpoint, individual paper, or weight artifact; the package's license and separately linked resources retain their own terms. |
| [sktime API](https://www.sktime.net/docs/api-reference/) | **First-party current time-series ML catalog enabled.** Structural links from its forecasting, transformation, classification, regression, clustering, alignment, detection, parameter-estimation, and pipeline pages retain every model-scoped class destination. The source-declared `#sktime.<module>.<Class>` fragment is an exact `sktime:machine-learning-component` identity while the fragment-free documentation URL is retained as the resource link. | Adds forecasting, classical and neural time-series classification/regression, clustering, anomaly/change-point detection, alignment, parameter estimation, transforms, and estimator composition without executing the package or using an estimator-name list. Shared documented class IDs merge across the nine task pages exactly. | This is a mutable current package API, not a model-release or checkpoint inventory. Third-party adapters and base/composition classes remain documented technique evidence; a card does not assert a paper, weights, or a trained artifact unless another source explicitly declares it. |
| [aeon API](https://www.aeon-toolkit.org/en/stable/api_reference/classification.html) | **First-party current time-series ML catalog enabled.** Its complete classification, regression, clustering, transformation, segmentation, and anomaly-detection task pages structurally enumerate each generated `aeon.<module>.<Class>` destination. The generated page's qualified path is the exact `aeon:machine-learning-component` identifier. | Adds current classical, hybrid, interval, shapelet, convolutional, recurrent, and other deep time-series components, plus transforms, segmentation, and anomaly-detection techniques, without package execution or a hand-maintained name list. Each documentation page remains model-scoped evidence. | This is a mutable package API, not a release/checkpoint catalog. A class can wrap a third-party dependency or be a base/composition component; it remains documented technique evidence until another source declares papers, weights, or a trained artifact. |
| [Darts forecasting models](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.html) | **First-party current forecasting-model index enabled.** Every class link whose source-declared `#darts.models.forecasting.<module>.<Class>` anchor is present becomes an exact `darts:machine-learning-component` identity; its fragment-free generated page is the model-scoped resource. | Adds baseline, statistical, regression-wrapper, neural, foundation, ensemble, conformal, and classifier forecasting techniques, including Darts' documented integrations for Chronos-2, TimesFM, TiRex, and PatchTST. | The mutable package page is a documentation source, not proof that an integration distributes a checkpoint or that the external model's paper/code/weights are licensed. Those links require independent source-declared evidence. |
| [PyOD detector inventory](https://pyod.readthedocs.io/en/latest/) | **First-party current anomaly-detection catalog enabled.** Every row in the document's Type/Abbr/Algorithm/Year/Class/Ref detector table is structurally selected; its qualified Class cell becomes the exact `pyod:machine-learning-component` identity and the linked modality API page is retained as model-scoped documentation. | Adds tabular, time-series, graph, text/image, audio, classical, ensemble, neural, and multimodal anomaly-detection components without package execution or a hand-maintained model list. Several documented algorithm variants can map to one qualified class; they remain aliases on the same implementation-backed component rather than invented releases. | This is a mutable documentation inventory, not a trained-model, paper, or weight registry. Individual citation anchors and declared external resources require source-specific enrichment before being attached as strong relations. |
| [PyGOD detector inventory](https://pygod.readthedocs.io/en/latest/) | **First-party current graph-anomaly detector catalog enabled.** Every generated `pygod.detector.<Class>` page structurally listed in its public index becomes an exact `pygod:machine-learning-component` identity, with duplicate navigation/table mentions reconciled by that source-native path. | Adds graph anomaly-detection components such as classical, matrix-factorization, autoencoder, GNN, GAN, contrastive, and generative detectors without package execution or a maintained model-name list. | This is mutable documentation evidence, not a checkpoint, paper, or artifact inventory. The source's model page is retained until an additional resource is expressly declared. |
| [Statsmodels API](https://www.statsmodels.org/stable/api.html) and retained [0.11.1 documentation](https://www.statsmodels.org/v0.11.1/api.html) | **First-party current plus retained historical statistical-model catalog enabled.** Structurally selected qualified class anchors from regression, generalized, discrete/count, multivariate, time-series, duration, robust, nonparametric, GAM, and miscellaneous model modules become exact `statsmodels:machine-learning-component` identities in the current API and 0.11.1, 0.12.2, 0.13.5, and 0.14.4 snapshots. | Adds statistical-learning and forecasting techniques including regression, GLM/GEE, discrete/count, factor/PCA, ARIMA/state-space, smoothing, vector, regime-switching, survival, and semiparametric models without package execution or a hand-maintained name list. Identical qualified IDs bridge versions exactly while distinct historical classes remain preserved. | This is documentation/component evidence, not trained releases or implied papers, checkpoints, or weights. Public retention begins at the currently available 0.11.1 snapshot; missing earlier hosted docs are not represented as source coverage. |
| [tslearn API reference](https://tslearn.readthedocs.io/en/stable/reference.html) | **First-party current time-series component catalog enabled.** Structural generated-page links select qualified class destinations from clustering, early classification, forecasting, matrix profile, neighbors, neural networks, piecewise transforms, shapelets, and SVM modules. Each full class path is an exact `tslearn:machine-learning-component` identity with its own model-scoped API page. | Adds time-series clustering, supervised/early classification, forecasting, nearest-neighbor, neural, shapelet, kernel/SVM, matrix-profile, and symbolic/piecewise techniques without package execution or a maintained name list. | This mutable API is documented technique evidence, not a pretrained-model, paper, code, or checkpoint inventory. Metrics, generators, datasets, and utility functions are intentionally outside this component scope. |
| [AutoGluon Tabular API](https://auto.gluon.ai/stable/api/autogluon.tabular.models.html) | **First-party current tabular model-component catalog enabled.** The source's exact `#autogluon.tabular.models.<Class>` anchors are structurally selected while adjacent method anchors are excluded; the fragment-free API page remains the model-scoped documentation resource. | Adds AutoGluon's documented tabular wrappers and native models, including tree, linear, nearest-neighbor, neural, multimodal, and tabular-foundation components, with exact qualified class identities and no package import or model-name list. | This is mutable package documentation, not a release/checkpoint catalog. The page does not establish that every class has an independently distributed paper, code repository, or weight artifact. A separately blocked endpoint is not represented as coverage. |
| [PyTorch Forecasting models](https://pytorch-forecasting.readthedocs.io/en/stable/models.html) | **First-party current forecasting component catalog enabled.** Each structurally linked generated `pytorch_forecasting.models.<module>.<Class>` page becomes an exact `pytorch-forecasting:machine-learning-component` identity; private auxiliary-package pages are excluded by the public path shape. | Adds documented DeepAR, N-BEATS/N-HiTS, recurrent, temporal-fusion, TiDE, TimeXer, and related neural forecasting components without importing the package or maintaining a class-name list. | This mutable API documentation is not evidence of a pretrained release, paper, code repository, or checkpoint unless the source declares those resources separately. |
| [GluonTS available models](https://ts.gluon.ai/stable/getting_started/models.html) | **First-party current row-scoped forecasting catalog enabled.** The document's exact `Model + Paper` / implementation matrix creates one `gluonts:machine-learning-component` entry for each source-declared model label. The leading model text is separated from an adjacent citation anchor, so author/year text never pollutes the entry identity. | Adds classical, state-space, neural, transformer, temporal-point-process, hierarchical, and wrapped forecasting techniques. Direct row-declared DOI/paper pages and GluonTS implementation files stay attached only to the matching model entry, yielding a useful paper + code + technique unit without a corpus download. | This is a mutable documentation matrix, not a complete historical catalog or a claim that all cited resources are downloadable. Rows without a declared citation retain only their exact implementation evidence; weights are not inferred. |
| [NeuralForecast models](https://nixtlaverse.nixtla.io/neuralforecast/models.html) | **First-party current neural-forecasting component catalog enabled.** Every structural `models.<slug>.html` destination becomes an exact `neuralforecast:machine-learning-component` identity; the source's explanatory label remains its display evidence. | Adds RNN/LSTM/GRU, transformer, convolutional, MLP/linear, N-BEATS/N-HiTS, DeepAR/DeepNPTS, graph, kernel, hierarchical, and modern forecasting architectures such as PatchTST, iTransformer, TimeMixer, SOFTS, and xLSTM. | This mutable documentation index does not itself assert a paper, implementation repository, release, or checkpoint for every listed architecture. Those resources require direct source declarations from the corresponding model page or another source. |
| [StatsForecast models](https://nixtlaverse.nixtla.io/statsforecast/) | **First-party current statistical-forecasting component catalog enabled.** Every source-declared bare lowercase fragment targeting the generated models API page becomes an exact `statsforecast:machine-learning-component` identity; the source's anchor text is retained as its display label. | Adds automatic, ARIMA, exponential-smoothing, theta, MSTL/TBATS, GARCH/ARCH, naive/window, Croston, intermittent-demand, and related classical forecasting techniques without importing the package or maintaining a model-name list. | This mutable API is documented technique evidence, not a trained-model/checkpoint or paper/code inventory. A source-anchor identity is stable within the published API but does not assert semantic equivalence to similarly named methods in other libraries. |
| [PyTorch Tabular models](https://pytorch-tabular.readthedocs.io/en/latest/models/) | **First-party current tabular component catalog enabled.** Each source-declared `#pytorch_tabular.models.<Class>Config` anchor becomes an exact `pytorch-tabular:machine-learning-component` identity while generic/module/method anchors are excluded structurally. | Adds documented category-embedding, AutoInt, DANet, GANDALF, tree-ensemble, NODE, TabNet, and stacking model components without importing the package or maintaining a model-name list. | This is mutable package documentation, not evidence that every configuration class supplies a pretrained release, paper, repository, or checkpoint. |
| [TorchGeo models](https://docs.torchgeo.org/en/stable/api/models.html) | **First-party current geospatial ML component catalog enabled.** Each structural `api/models/<slug>.html` destination becomes an exact `torchgeo:machine-learning-component` identity, including generic and geospatial-specialized model families. | Adds geospatial foundation/pretraining, remote-sensing localization, segmentation, atmospheric, temporal, and change-detection components including SatCLIP, CROMA, DOFA, OlmoEarth, Copernicus-FM, change-detection architectures, and standard vision backbones. | This mutable model-family index is not a release/checkpoint inventory. Individual pages can later contribute their direct papers, implementation, and weights; this catalog does not infer them. |
| [Segmentation Models PyTorch](https://smp.readthedocs.io/en/latest/models.html) | **First-party current segmentation architecture catalog enabled.** Exact top-level `#segmentation_models_pytorch.<Class>` anchors become `segmentation-models-pytorch:machine-learning-component` identities; implementation-specific decoder internals do not match. | Adds U-Net/U-Net++, FPN, PSPNet, DeepLabV3/V3+, Linknet, MAnet, PAN, UPerNet, Segformer, and DPT segmentation architectures without importing the package or maintaining a model-name list. | This mutable API is architecture documentation, not a claim that the package distributes a distinct paper, release, checkpoint, or model card for every class. |
| [torchvision](https://docs.pytorch.org/vision/main/models) and its [source-declared weight enums](https://github.com/pytorch/vision/tree/main/torchvision/models) | **Official family pages plus static `WeightsEnum` release expansion are enabled.** One pinned source archive is AST-parsed without importing TorchVision; every literal `Weights(url=...)` enum member becomes a versioned released weight observation tied to its exact enum-class model identity. | Adds current documented vision families and their source-declared classification, detection, segmentation, optical-flow, video, quantized, and other official pretrained releases, including exact weight URLs and pinned model-definition provenance. | A weight enum is a TorchVision release identity, not proof of a paper, upstream origin, or relationship to a similarly named external checkpoint. Source code and weights can have different licenses; weight files are reference-only and never downloaded. |
| [PyTorch Hub](https://pytorch.org/hub/) | **First-party research-model page catalog enabled.** Every structurally listed `/hub/<slug>` card becomes a source-native `pytorch:hub-model-page` identity with its own model-scoped Hub page. A bounded resolver retains only direct GitHub, paper, Hugging Face, and checkpoint references from the card, never its navigation. | Adds curated research implementations across vision, NLP, speech, audio, video, generation, and classical architectures, with source-declared reproducibility links ready to merge into the same exact Hub-card entry. | A Hub slug is not equivalent to a GitHub repository, a framework builder, an origin paper, or a checkpoint. The mutable beta catalog does not establish a complete historical census; the source retains no code or weight bytes. |
| [GluonCV Model Zoo](https://cv.gluon.ai/model_zoo/index.html) | **Historical first-party released-model tables enabled** for classification, detection, segmentation, pose, action recognition, and depth. Every structural row becomes a task-specific `gluoncv:*:model` handle; Sphinx footnote markers and display-only input-size suffixes are stripped structurally, not treated as model names. Row-declared training recipes/logs/configs are scoped to that model, while the documented GluonCV repository is a collection-level implementation resource. | Restores older MXNet/GluonCV pretrained evidence across major vision tasks, including released model handles and any exact recipe/log/config URL the tables provide. The tables' hashtag/checksum-like fields are retained only as source evidence; no weight URL is inferred from them. | This is the maintained 0.11 documentation plane, not a full historical export or an assertion that every listed artifact remains downloadable. Recipe/log/config files and weights are never bulk downloaded. |
| [Ultralytics models](https://docs.ultralytics.com/models/) | **Current and historical first-party family matrix enabled.** Each Model/Tasks/Modes row must supply a canonical model-document URL; presentation-only “NEW” labels are structurally stripped before forming the source-native ID. | Adds directly documented YOLO generations, SAM, MobileSAM, FastSAM, YOLO-NAS, RT-DETR, YOLO-World, and YOLOE family evidence with a scoped source page for each. | This is family-level documentation, not an individual checkpoint/version inventory. Per-family pages can be resolved later; weights are not inferred or downloaded from the matrix. |
| [Torchaudio models](https://docs.pytorch.org/audio/stable/models.html), [pretrained pipelines](https://docs.pytorch.org/audio/stable/pipelines.html), and [pipeline source](https://github.com/pytorch/audio/tree/main/src/torchaudio/pipelines) | **Official model-class and pretrained-pipeline catalogs plus a static pipeline-definition registry enabled.** The static reader resolves one pinned source archive, recognizes only the initializer's all-caps bundle exports, and retains each literal checkpoint path, direct model-specific docstring resource, and pinned definition. | Adds documented speech, audio-generation, source-separation, and self-supervised audio architectures; exact public pipeline identities; checkpoint references; and source-declared training-code, license, or demo links when present. The registry uses the same `torchaudio:pipeline` identity as the documentation catalog, so these facts enrich the corresponding entry rather than invent a parallel model. | Package documentation is distinct from the licenses for cited data, model weights, and third-party implementations. Citation keys without an explicit source URL are not treated as paper links. Weight archives are reference-only and never downloaded. |
| [Keras Applications](https://keras.io/api/applications/), [KerasHub preset families](https://keras.io/keras_hub/presets/), and [KerasHub source declarations](https://github.com/keras-team/keras-hub/tree/master/keras_hub/src/models) | **Official Applications table, KerasHub family directory, and static source-declared preset registry enabled.** One pinned KerasHub archive is AST-parsed; every literal provider-backed preset becomes an exact source-native entry with its pinned definition and exact Kaggle/Hugging Face handle. | Applications rows and individual KerasHub presets are released catalog assertions; KerasHub family rows remain `documented` evidence. Provider pages and source definitions are model-scoped, while weight bytes are never fetched. | A KerasHub preset is not inferred to be its architecture's original paper or a universal checkpoint identity. Follow each artifact's license and host limits. |
| [PyTorch Geometric models](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html) | **Official model-class directory enabled.** A structural link rule selects every class page in the `torch_geometric.nn.models` API namespace and excludes adjacent helper functions. | Supplies documented graph, molecular, and geometric model-family evidence with stable API-page identifiers. Model weights, checkpoints, and publication mappings remain follow-on work. | Library documentation is not a grant for separately distributed checkpoints; preserve each artifact's declared license. |
| [TorchDrug models](https://torchdrug.ai/docs/api/models.html) | **Official model-class directory enabled.** The extraction rule pairs every model display heading with the immediately following canonical `torchdrug.models.*` class anchor. | Supplies documented graph, molecular, protein, generative, and knowledge-graph model-family evidence. It does not claim that package classes imply an available checkpoint. | Documentation does not grant rights in separately distributed data or weights; preserve declared rights when artifacts are linked. |
| [Transformers](https://github.com/huggingface/transformers/blob/main/docs/source/en/_toctree.yml) | **The official model-document navigation manifest is enabled;** versioned auto-config introspection remains. | Captures every current model-doc entry and its stable document slug. Hugging Face Hub separately enumerates public checkpoints. | Library support is not proof that every checkpoint is licensed or still available. |
| [PaddleNLP transformer families](https://paddlenlp.readthedocs.io/en/latest/model_zoo/) | **The official task-support matrix is enabled.** Each structural Model row becomes a `paddlenlp:transformer-family` entry, and its direct arXiv destination, when declared in that row, is retained as a scoped paper reference. | Adds first-party documentation evidence for PaddleNLP's supported transformer families and source-declared paper connections without broad name matching. Rows without a paper link stay paperless. | This is a family/documentation plane, not a checkpoint catalog or a claim that the matrix covers historical releases. No paper or artifact bytes are fetched. |
| [PaddleNLP pretrained embeddings](https://paddlenlp.readthedocs.io/en/latest/model_zoo/embeddings.html) | **The official embedding matrix is enabled.** Each structurally valid Word2Vec, GloVe, or fastText cell becomes a `paddlenlp:embedding` entry; explicit `N/A` cells are skipped by a configured identifier pattern. | Adds exact PaddleNLP loadable embedding handles across Chinese and English corpora, including distinct target/context and dimensionality variants. It retains the complete source document as per-entry provenance. | The page does not declare stable artifact URLs for these cells, so no weight download is inferred. This source says the identifiers are documented and loadable through PaddleNLP, not that each has a separately archived, current checkpoint. |
| [Sentence Transformers pretrained models](https://www.sbert.net/docs/sentence_transformer/pretrained_models.html) | **The maintained curated repository document is enabled.** Only concrete Hugging Face owner/repository destinations are selected; model name and identity both derive from that exact path, while generic Hub search, papers, Spaces, and blog URLs are excluded. | Adds curated semantic-search, multilingual, multimodal, INSTRUCTOR, and scientific-similarity model-card evidence. Its `huggingface:model` identifier directly converges with the general public Hub enumerator when both observe the same repository. | This is a curated documentation plane, not a complete inventory of all community Sentence Transformers models. The individual Hub model cards are references for later bounded enrichment; they are not bulk fetched by this catalog source. |
| [spaCy pipeline compatibility](https://github.com/explosion/spacy-models/blob/master/compatibility.json) | **The official complete compatibility manifest is enabled.** Each pipeline package becomes a `spacy:model` entry and every listed package version a `spacy:package-version` release, retaining the series that declares compatibility. | Adds current and retained historical trained language-pipeline packages across spaCy's published languages and architectures. The manifest is a compact source of exact package/version evidence rather than a name-matched collection. | It does not guarantee that every historical wheel remains installable or supply stable per-wheel URLs. The adapter references the manifest and never derives or downloads binary locations. |
| [Stanza resources](https://github.com/stanfordnlp/stanza-resources) | **Every retained versioned resource manifest is enabled at one public Git revision.** Every language/processor/package item with a declared MD5 checksum becomes an exact `stanza:model` and a versioned release observation; source-declared dependencies become `depends_on` relations. | Adds historical and current pretrained NLP processor releases across Stanza's language resources, retaining source checksums, manifest revision, and package dependencies. The adapter does not use a model-name list or cross-source fuzzy matching. | The manifests prove released resource entries and checksums, not continuing availability of model files. They do not declare stable per-file URLs, so weights are neither synthesized nor downloaded. |
| [DeepChem model cheatsheet](https://deepchem.readthedocs.io/en/latest/api_reference/models.html) | **Official general, molecular, and materials inventory enabled.** A table rule selects all Model/Reference rows from the complete first-party cheatsheet. | Supplies documented scientific model-family evidence and complete row context. Individual model-page links are not required and no checkpoint or paper relationship is inferred from the row alone. | Package documentation does not grant rights in datasets, cited papers, or separately distributed weights. |
| [fastText Common Crawl vectors](https://fasttext.cc/docs/en/crawl-vectors.html) | **Every structurally matched language cell is enabled.** Each first-party row must declare matching `cc.<language>.300.bin.gz` and `.vec.gz` files; both stay scoped to that exact language release. The catalog's own paper, fastText implementation, and license anchors are retained as shared source-declared resources. | A current verification found 158 structurally valid language/vector pairs, although the document heading and prose say “157 languages”; the source preserves every explicit matching row rather than silently discarding one. Each entry carries the `fasttext:cc-vector` language handle, source catalog page, two reference-only weight URLs, and declared shared resources. | The page declares CC BY-SA 3.0 for the vectors. Vector bytes are not fetched, and the source makes no claim about alternative fastText releases or a complete historical catalog. |
| [Diffusers](https://huggingface.co/docs/diffusers/api/pipelines/overview) | **The official available-pipelines table is enabled;** versioned class/config expansion remains. | Captures documented pipeline families and techniques, while Hub ingestion supplies hosted repositories. | Many scientific diffusion implementations do not use Diffusers; domain corpora below are required. |
| [ONNX Model Zoo](https://github.com/onnx/models) | **The complete public validated-model index is enabled as a historical source.** Every linked `validated/` entry becomes a distinct model artifact with its repository path and row-level paper/external artifact references. | Captures classic and deployed ONNX model releases across vision, language, audio, and other domains without downloading binaries. The Zoo's retirement is retained as historical context rather than treated as current hosting. | The repository index is Apache-2.0, but linked paper and model artifacts retain their own rights; model binaries are referenced, never mirrored. |
| [Apple Core ML Model Gallery](https://developer.apple.com/machine-learning/models/) | **The public first-party package table is enabled.** Every structurally matched `Model Name / Size / Action` row becomes an exact `apple:coreml-model-package` identity; its direct `ml-assets.apple.com` download stays scoped to that one row as a non-crawled `weights` reference. | Adds released Core ML conversion/inference packages across vision, depth, segmentation, language, and other Apple-published examples without downloading package bytes. | A Core ML package filename is a source-native release identity, not an assertion that it is the upstream technique or original checkpoint. Page-level code or paper links outside a row are deliberately not attributed to every package. |
| [TensorFlow Detection, DeepLab, Slim, and Model Garden release tables](https://github.com/tensorflow/models/tree/master/research/object_detection/g3doc) | **Five maintained first-party release tables are enabled at a pinned repository commit.** TF1/TF2 Detection `Model name` rows, DeepLab's explicit `Checkpoint name`/`Model name` rows, TF-Slim's `Model` rows, and Model Garden NLP `Model` rows must each declare a direct artifact before becoming released source-native entries. | Adds TensorFlow 1 COCO, Kitti, Open Images, iNaturalist, AVA, Serengeti, and mobile/Edge-TPU detector releases; TensorFlow 2 CenterNet, EfficientDet, SSD, Faster R-CNN, Mask R-CNN, and related detectors; DeepLab PASCAL VOC, Cityscapes/ADE20K, and ImageNet/COCO-pretraining releases; TF-Slim ImageNet classifiers such as Inception, ResNet, NASNet, MobileNet, and PNASNet; and Model Garden BERT, ALBERT, ELECTRA, and other explicitly released NLP variants. Every admitted row preserves its exact source-declared artifact as a reference-only resource. | TF generations, tasks, and release identities remain separate unless an exact source identifier bridges them. The header rule is source-specific: it does not treat unrelated tables in the documentation as models. These catalogs do not cover arbitrary TensorFlow repository code, derived conversions, or imply origin-paper links. Archives and model artifacts are never downloaded. |
| [TensorFlow TPU model zoo](https://github.com/tensorflow/tpu/blob/master/models/official/detection/MODEL_ZOO.md) | **Every direct-artifact row in the first-party lower-case `model` tables is enabled at a pinned commit.** The document spans TensorFlow TPU object detection, instance segmentation, and image-classification releases; prose and rows lacking an explicit artifact stay out. | Adds RetinaNet, Mask R-CNN, ResNet, SpineNet, NAS-FPN, SpineNetMB, and related checkpoint evidence with the exact experiment-table heading, model label, first-party source document, repository, and reference-only artifact. | This is a retained project release table, not an assertion that one repeated model label combines every training run into a universal architecture identity. No checkpoint, config, or artifact link is fetched. |
| [timm model registry](https://huggingface.co/docs/timm/en/quickstart) | **Every architecture explicitly imported by `timm.models` and registered with `@register_model` is enabled.** One bounded GitHub source archive is resolved at a public commit; it is parsed statically, never imported or executed. | Adds exact timm architecture IDs, deprecated aliases, pinned definition code, and literal pretrained configurations. The registry's `timm/` Hub shorthand is expanded exactly as upstream does to `timm/<architecture>.<tag>`; direct Hub, external-weight, and license URLs remain distinct references. | Source code follows the repository license. A configuration can document a loading location but does not prove a current checkpoint, a paper, or separate artifact rights; no checkpoint bytes are fetched. |
| [PaddlePaddle ModelCenter](https://github.com/PaddlePaddle/models) | **Every public `modelcenter/*/info.yaml` family manifest and its source-declared Chinese/English download-table variants is enabled.** The adapter resolves a public commit, rejects a truncated tree, and reads only bounded text manifests. | Adds source-native Paddle model and release identities, family/model-card/configuration documentation, named Paddle project implementation, explicit papers, tasks, and direct inference/pretrained artifact references. A family without a download table remains documented rather than fabricated as a checkpoint release. | This is ModelCenter coverage, not a claim that it exhausts the broader `PaddlePaddle/models` repository or its historical “600+” statement. Follow each linked artifact's terms; no model archive is transferred. |
| [PaddleClas model registry](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.6/paddleclas.py) | **Every literal ImageNet-series and PULC model paired with its registry's literal archive template is enabled.** The adapter pins one source revision and AST-parses those declarations without importing PaddleClas; its ShiTu input label is deliberately excluded because the same source expands it into different component downloads rather than a one-to-one archive. | Adds source-native PaddleClas model/release IDs, series-family context, pinned definition/repository provenance, and one direct reference-only inference archive per admitted model. | This covers only the command-line source's explicitly paired ImageNet/PULC inventories, not every PaddleClas architecture, recipe, conversion, historical revision, or archive's continuing availability. Archive bytes are never fetched. |
| [PaddleDetection model zoo](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.9/configs) | **Every direct checkpoint URL in a `configs/**/README.md` table whose header declares a download/weight/checkpoint column is enabled.** One bounded repository archive is pinned and parsed as Markdown only. Rows without an explicit known checkpoint file URL are not inferred. | Adds source-native checkpoint-model and checkpoint-release IDs, pinned model-zoo document provenance, row context, first-column aliases, explicit direct checkpoint references, and source-declared relative configuration files scoped to that checkpoint. Identical direct checkpoint URLs observed in multiple tables retain every supporting row. | This is the versioned config-document release plane, not a claim that all PaddleDetection source configs had public checkpoints or remain available. The adapter does not execute code, read arbitrary prose tables, or transfer checkpoint bytes. |
| [Paddle3D model zoo](https://github.com/PaddlePaddle/Paddle3D/tree/develop/docs/models) | **Every direct checkpoint URL in an admitted first-party `docs/models/**/README.md` model-zoo table is enabled under its own `paddle3d:*` identifiers.** One bounded archive is pinned and structurally parsed; the root repository's broader architecture menu is not treated as checkpoint evidence. | Adds 3D perception, lidar detection/segmentation, monocular detection, BEV-camera, and fusion release evidence with exact documentation, row-local architecture aliases, source-declared configuration, and reference-only model URLs. | This document-family plane covers only direct recognized checkpoint links, not every listed Paddle3D technique, source config, or currently reachable artifact. It neither executes source code nor downloads model bytes. |
| [PaddleGAN tutorial model zoo](https://github.com/PaddlePaddle/PaddleGAN/tree/develop/docs/en_US/tutorials) | **Every direct first-party file under PaddleGAN's declared `/models/` download path in its English tutorials is enabled at one pinned commit.** Download-table links remain on their exact row; tutorial-level paper, official-code, or hosted-example links are used only where that page declares one model artifact. | Adds GAN, image translation/restoration, face enhancement, motion, and video-super-resolution checkpoints with exact tutorial/repository provenance, direct reference-only model files, row-local papers, and unambiguous single-model paper/code/demo references. | This source deliberately excludes dataset links, third-party weight guesses, Chinese tutorials, and ambiguous multi-model document references. It does not follow links, execute code, or transfer artifact bytes. |
| [PaddleSeg model zoo](https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.10/configs) | **Every direct checkpoint URL in an admitted `configs/**/README.md` model table is enabled under its own `paddleseg:*` identifiers.** The same pinned archive reader accepts the project's compact `:-:` separators and escaped table-cell delimiters while still requiring a direct recognized checkpoint-file URL. | Adds source-native segmentation checkpoint/release identities, exact model-zoo documentation, first-column architecture aliases, direct reference-only checkpoint URLs, and any source-declared configuration paths. It retains repeated supporting rows without conflating PaddleSeg and PaddleDetection entries. | This is an explicit-release table plane, not a claim that every PaddleSeg configuration has weights, that every URL remains available, or that same-named architectures are the same release. No code or artifact bytes are fetched. |
| [PaddleVideo model zoo](https://github.com/PaddlePaddle/PaddleVideo/tree/develop/docs/en/model_zoo) | **Every direct checkpoint URL in the maintained English `docs/en/model_zoo/**/*.md` table set is enabled under separate `paddlevideo:*` identities.** The pinned archive reader is restricted to this source-defined page family and accepts an entry only when a table header declares weights/downloads and the row gives a recognized direct checkpoint file. | Adds action-recognition, localization, detection, estimation, multimodal, partition, and video-segmentation release evidence with exact model-zoo documentation, first-column aliases, direct reference-only weights, and any declared pinned configuration link. | English model-zoo pages are the configured scope, not every PaddleVideo example, Chinese page, arbitrary prose link, or historical release. The source never executes code or fetches checkpoint bytes. |
| [ESPnet Model Zoo](https://github.com/espnet/espnet_model_zoo) | **Every structurally valid public `table.csv` record is enabled.** The Model Zoo's own query/download tooling consumes this small first-party table; one commit lookup plus one bounded CSV fetch produces an immutable catalog observation. | Adds exact ESPnet model and release IDs, task/corpus/sample-rate/language/framework metadata, direct source table and code links, and source-declared Zenodo or Hugging Face artifact references. Provider-invalidated rows are retained with an explicit validation flag for historical fidelity. | A listed archive URL is reference evidence, not an availability or rights guarantee. The adapter retains no archive bytes and does not infer a paper relation from a model name. |
| [Fairseq wav2vec, wav2vec 2.0, and translation release tables](https://github.com/facebookresearch/fairseq/blob/main/examples/wav2vec/README.md) | **Every row under each configured first-party `Description`/`Model` header that has a same-row direct artifact is enabled at a pinned commit.** The original wav2vec and wav2vec 2.0 selectors read their shared README independently; translation uses its own source document. | Adds historical self-supervised speech and neural-machine-translation releases with exact Fairseq source identifiers, row-level corpus and fine-tuning context, source documentation/repository, and reference-only checkpoints. | Model names, tables, or files absent from these explicitly scoped documents are not inferred. Same text in the two wav2vec tables does not merge identities; artifacts are neither followed nor downloaded. |
| [Omnilingual ASR model cards](https://github.com/facebookresearch/omnilingual-asr/blob/main/src/omnilingual_asr/cards/models/rc_models_v2.yaml) | **Every named, top-level card with a literal direct checkpoint URL is enabled at a pinned commit.** The narrow line reader accepts only `name`, optional model-family/architecture, and `checkpoint` fields; it does not load YAML or evaluate source text. | Adds source-native multilingual ASR release handles with source-declared family/architecture context, exact card provenance, the official repository, and reference-only checkpoint URLs. | Tokenizer and incomplete cards are excluded. Nested or dynamic YAML, undocumented artifacts, paper relations, and external files are not inferred, followed, or downloaded. |
| [Seamless communication release tables](https://github.com/facebookresearch/seamless_communication/blob/main/README.md) | **Every named row under a first-party `Model Name` header that includes a same-row direct checkpoint is enabled at a pinned commit.** The independent SeamlessM4T, streaming, and W2v-BERT tables remain source-scoped rows, even where their artifacts are hosted on Hugging Face. | Adds multilingual translation/ASR release identities with first-party model-card, checkpoint, and row-scoped metrics links. Metrics archives retain an evaluation-evidence relation rather than being mislabeled as weights. | Gated Expressive models and prose-only models are excluded. Page-level papers are not broadcast to individual releases, and artifacts are never followed or downloaded. |
| [SONAR text-model download table](https://github.com/facebookresearch/SONAR/blob/main/README.md) | **Every source-declared `encoder`, `decoder`, or `finetuned decoder` row with a direct checkpoint is enabled at a pinned commit.** The model-name selector rejects the neighboring tokenizer resource rather than treating it as a model. | Adds exact multilingual text-embedding and translation component releases with source-table and repository provenance plus reference-only direct weights. | The broad language lists and speech-encoder language files do not become per-language model assertions; tokenizer and unknown table rows are excluded. No artifacts are followed or downloaded. |
| [Segment Anything](https://github.com/facebookresearch/segment-anything/blob/main/README.md), [SAM 2](https://github.com/facebookresearch/sam2/blob/main/README.md), and [ImageBind](https://github.com/facebookresearch/ImageBind/blob/main/README.md) release tables/lists | **Every configured first-party row or explicitly headed list item with a named model and same-row direct checkpoint is enabled at a pinned commit.** Original SAM admits only its source-native `vit_*` list handles; SAM 2’s selector captures only its declared `sam2`/`sam2.1` Hiera prefix before presentation-only config/checkpoint labels; ImageBind retains its published `imagebind_huge` row. | Adds promptable image/video segmentation and six-modality embedding releases with exact source-table/list provenance, source repository, direct reference-only artifacts, and model-scoped source identifiers. | The list reader does not turn arbitrary README bullets into models. Neither selector broadcasts README prose, demonstration resources, or page-level papers to a per-release relationship. No checkpoint is followed or downloaded. |
| [DETR](https://github.com/facebookresearch/detr/blob/main/README.md) and [ConvNeXt](https://github.com/facebookresearch/ConvNeXt/blob/main/README.md) release tables | **Every first-party `name` row with a same-row direct model artifact is enabled at a pinned commit.** DETR’s backbone and ConvNeXt’s resolution are retained as source-row identity context, so repeated display names do not collapse distinct checkpoints. | Adds end-to-end detection/panoptic and modern ConvNet classification release evidence with exact table provenance, direct reference-only checkpoints, and source repository. DETR's same-row logs retain a training-log relation. | README-level papers, notebooks, and unrelated prose are not broadcast to release rows. Log files are resources, not weights, and no URL is followed or downloaded. |
| [Swin Transformer release tables](https://github.com/microsoft/Swin-Transformer/blob/main/README.md) | **Every named first-party V1/V2/SwinMLP row with a same-row direct model artifact is enabled at a pinned commit.** The declared pretraining field distinguishes the ImageNet-1K and ImageNet-22K release lines. | Adds original Swin, SwinV2, and SwinMLP classification/pretraining releases with model-card table provenance, direct GitHub/Baidu artifacts, source configuration links where stated, and source-declared logs. | Links in external comparison rows remain row-local evidence, not an equivalence claim. Page-level papers are not broadcast, and no checkpoint/log URL is followed or downloaded. |
| [DeiT](https://github.com/facebookresearch/deit/blob/main/README_deit.md), [CaiT](https://github.com/facebookresearch/deit/blob/main/README_cait.md), [ResMLP](https://github.com/facebookresearch/deit/blob/main/README_resmlp.md), and [PatchConvNet](https://github.com/facebookresearch/deit/blob/main/README_patchconvnet.md) release tables | **Every configured first-party `name` row with a same-row direct checkpoint is enabled at a pinned commit.** Each document keeps its source-native family namespace; CaiT and PatchConvNet retain the declared resolution context where a visible variant name repeats. | Adds historical data-efficient vision transformer, class-attention transformer, MLP, and patch-convolutional releases with source-document/repository provenance and direct reference-only artifacts. | Similar variant text across documents is not a merge key. Top-level paper citations and prose are not broadcast to model rows, and artifacts are not followed or downloaded. |
| [PySlowFast](https://github.com/facebookresearch/SlowFast/blob/main/MODEL_ZOO.md) and [TimeSformer](https://github.com/facebookresearch/TimeSformer/blob/main/README.md) release tables | **Every named first-party architecture/name row with a same-row direct artifact is enabled at a pinned commit.** PySlowFast retains the publisher's size context across action-recognition/detection and video-transformer families; TimeSformer retains its declared source dataset. | Adds historical video-classification, action-detection, multiscale/reversible transformer, and TimeSformer release evidence with direct reference-only checkpoints and pinned source/repository provenance. | Repeated source names collect only their source-declared artifacts; names never merge across these projects. `.pkl` is classified as a checkpoint format, not a generic model resource. No artifacts are followed or downloaded. |
| [PaddleSlim Model Zoo](https://github.com/PaddlePaddle/PaddleSlim/blob/develop/docs/en/model_zoo_en.md) | **Every named row in the first-party English model-zoo tables that declares a direct artifact is enabled at a pinned commit.** The parser requires a model column and an artifact-bearing source row, rather than treating compression documentation or model names in prose as releases. | Adds compression, distillation, pruning, quantization, NAS, and related exact source-native model/release identities, with the table row, repository, model-zoo document, and direct reference-only artifact link. | A PaddleSlim variant is not presumed to be the same entry as its upstream architecture. Rows without a direct artifact and unrelated tables are excluded; model files are never fetched. |
| [PaddleRec support table](https://github.com/PaddlePaddle/PaddleRec/blob/master/README_EN.md) | **Every row in the first-party `Type / Algorithm / Online Environment / Parameter-Server / Multi-GPU / version / Paper` table is enabled at a pinned commit.** The first source-declared Algorithm link names the implementation; same-row model documentation, hosted examples, versions, and paper links are retained without cross-row broadcasting. | Adds recommendation techniques across content understanding, matching, recall, ranking, multi-task, and re-ranking with exact category-plus-algorithm identities, first-party implementation/doc links, and source-declared paper references. | This is a documented-technique inventory, not a checkpoint catalog: it creates no release or artifact URL that the table does not state. The parser excludes all other README tables, does not follow its links, and never downloads source or model bytes. |
| [PaddleSpeech released models](https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/docs/source/released_model.md) | **Every row with a declared neural-model heading, model cell, and direct artifact link is enabled.** The adapter parses the first-party table at one commit and excludes the document's explicitly non-neural N-gram language-model section. | Adds speech recognition, self-supervised speech, speech translation, TTS, vocoder, audio-classification, speaker-verification, and punctuation-model entries with direct model/checkpoint/inference links and task/dataset context. | A table entry is direct source evidence only. Relative example links and rows without an artifact are not promoted; no checkpoint is downloaded. |
| [PaddleOCR V2 model list](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version2.x/ppocr/model_list.en.md) | **Every first-party downloadable-model row in the versioned V2 list is enabled.** Repeated source rows for one model/table identity retain the union of their explicitly declared artifacts. | Adds OCR detection, recognition, multilingual, classification, and mobile deployment releases with pinned documentation and inference/trained/pretrained/ONNX/Paddle-Lite references. | This is a historical V2 documentation plane, deliberately separate from newer PaddleOCR pages. It does not claim current availability and never transfers an artifact. |
| [PaddleOCR current model list](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/model_list.md) | **Every current HTML-table row whose first header is `模型`/`模型名称`, which explicitly declares `模型下载链接`, and which gives a recognized direct artifact file is enabled at a pinned commit.** A row's model, weights, YAML configuration, and other direct resources stay scoped to that same source row. | Adds current PaddleOCR model/release identities, pinned source/repository provenance, direct reference-only trained/inference artifacts, explicit YAML configuration, and any other source-declared row resource. Identical exact model names retain all supporting rows without promoting an empty or malformed link. | This current-document plane is separate from V2 identities; it does not infer missing artifacts, scrape prose, execute code, follow linked pages, or fetch artifact bytes. |
| [Detectron2 Model Zoo map](https://github.com/facebookresearch/detectron2/blob/main/detectron2/model_zoo/model_zoo.py) | **Every literal entry in the source's map expressly described as officially released pre-trained Detectron2 models is enabled.** The adapter resolves one public commit and AST-parses the static mapping only; it never executes upstream Python. | Adds exact source-native config and release identifiers, a pinned source/repository reference, the corresponding checked-in configuration, and the direct official checkpoint URL. | The source only covers the model-zoo map, not external Detectron2 projects or inferred papers. Configuration and checkpoint URLs are reference-only and not fetched. |
| [CenterNet, CenterTrack, AlphaPose, pycls, VISSL, MaskFormer, Mask2Former, legacy Detectron, detrex, Detic, and X-AnyLabeling model zoos](https://github.com/xingyizhou/CenterNet/blob/master/readme/MODEL_ZOO.md) | **Each first-party historical table is enabled only for visible configured-model rows that declare a same-row direct artifact.** Independent sources retain their own pinned commit, table heading, and namespace rather than merging a shared display label; VISSL additionally preserves source-declared method and pretraining-corpus context, while legacy Detectron's `^{_{…}}` display-cell wrapper is normalized before applying its strict backbone/download rule. | Adds released object detection, tracking, monocular 3D tracking, human/whole-body pose, image-classification, self-supervised representation-learning, universal-image-segmentation, DETR-family, open-vocabulary-detection, and deployment-oriented ONNX evidence, with direct Google/Baidu/official artifact references where declared and source implementation provenance. | These compact research-project tables are not general package registries or origin-paper assertions. Artifacts remain links only; the reader neither follows host pages nor downloads weights. |
| [DINO and DINOv2 pretrained-model tables](https://github.com/facebookresearch/dino/blob/main/README.md) | **Every visible `arch`/`model` table row that declares a same-row direct checkpoint is enabled at a pinned commit.** The documents are read independently, preserving their source table heading and exact released model label rather than equating variants by backbone text. | Adds first-party DINO and DINOv2 self-supervised vision releases, including source-declared `.pth` and ONNX weights where present, with pinned documentation and model-scoped implementation provenance. | Only direct artifact rows become releases. Training prose, model-family mentions, access-controlled links, and evaluation-only table resources are not inferred into additional models; no source or model bytes are downloaded. |
| [MAE pretrained checkpoint table](https://github.com/facebookresearch/mae/blob/main/README.md) | **Every named column in the exact first-party `pre-trained checkpoint` row is enabled at a pinned commit.** This document's model variants are columns rather than rows; the reader transposes only the configured checkpoint row and requires a direct recognized artifact in each cell. | Adds released MAE ViT Base, Large, and Huge pretraining artifacts with their source-specific model labels, pinned documentation, and official implementation provenance. | Metrics, MD5-only cells, fine-tuning instructions, and other table rows do not become releases. This source preserves the original column layout as model-scoped link evidence and never transfers checkpoint bytes. |
| [ESM pretrained models](https://github.com/facebookresearch/esm/blob/main/README.md) | **Every visible row whose source header is exactly `esm.pretrained.` and which declares a same-row direct artifact is enabled at a pinned commit.** The model-local loader shorthand and training corpus remain source-row scoped. | Adds ESM protein-language-model and structure-model releases with literal first-party names, direct reference-only checkpoint URLs, source documentation, and corpus context. | The source is a published release table, not evidence that all ESM-related research objects are equivalent or every discussed model has a direct artifact. No checkpoint or source byte is downloaded. |
| [OpenVINO Open Model Zoo](https://github.com/openvinotoolkit/open_model_zoo) | **Every public `models/*/*/model.yml` manifest is enabled.** The adapter resolves one Git commit, validates that GitHub's recursive tree is complete, and reads each bounded manifest without downloading model files. | Adds maintenance-mode historical/production releases across vision, language, speech, and other tasks. Each entry retains its exact OpenVINO identity, task/framework/description, manifest and README, declared files/checksums/license, and original source URL; a source-declared Hugging Face origin becomes a `converted_from` edge rather than a fuzzy merge. | The repository is Apache-2.0, but each model/source file has its declared license and access terms. File URLs are reference-only and are never fetched as model bytes. |
| [OpenMMLab model indexes](https://github.com/open-mmlab/mmdetection/blob/main/model-index.yml) | **The complete MMDetection, MMPose, MMSegmentation, final MMClassification 0.x, MMPreTrain, MMAction2, MMDetection3D, MMOCR, MMRotate, MMYOLO, MMagic, MMRazor, MMTracking, MMSelfSup, MMFewShot, MMFlow, and MMGeneration manifest graphs are enabled.** Each source resolves a public commit, imports every listed metafile at that exact revision, and admits only source-declared model/config/weight rows. | Creates separate artifact records for individual vision, video, OCR, 3D, generation, compression, detection, pose, segmentation, tracking, self-supervised, few-shot, and optical-flow configurations, with version-pinned configuration, checkpoint, code, README, and paper references where the manifest declares them. | Each project and weight URL has its own license and retention status. Historical project indexes remain independent from successor projects; the registry retains metadata and links only, and does not download checkpoint bytes. |
| [MMEngine runtime checkpoint maps](https://github.com/open-mmlab/mmengine/tree/main/mmengine/hub) | **Every literal handle-to-direct-checkpoint row in MMEngine's maintained `openmmlab.json` and `mmcls.json` maps is enabled at one pinned commit.** The reusable reader rejects arrays, nested objects, relative URLs, and non-checkpoint file targets rather than inferring a release from package code. | Adds exact source-native pretraining handles, direct reference-only weight files, the pinned map as model-card evidence, and the official runtime repository as source implementation. Each JSON mapping row becomes one model-scoped release. | These maps are runtime artifact registries, not a statement that similarly named models across OpenMMLab projects are identical, nor paper evidence. No package import, artifact follow, or checkpoint transfer occurs. |
| [OpenAI CLIP and Whisper checkpoint maps](https://github.com/openai/CLIP/blob/main/clip/clip.py) | **Every row in each source's sole literal `_MODELS` dictionary is enabled at one pinned commit.** The AST reader requires literal unique string handles and direct recognized checkpoint-file URLs, rejecting expressions, non-map assignments, and aliases instead of executing upstream code. | Adds exact CLIP vision-language and Whisper ASR model handles with their direct reference-only `.pt` weights, source file, and official source implementation. Each literal mapping row becomes a separate model-scoped release, even when the publisher declares an alias to the same weight file. | These maps are package runtime registries, not source-declared paper or architectural-equivalence evidence. The reader neither imports their Python, follows artifacts, nor downloads checkpoint bytes. |
| [MMHuman3D model zoo](https://github.com/open-mmlab/mmhuman3d/blob/main/docs/model_zoo.md) | **Every direct recognized checkpoint URL in a first-party `configs/**/README.md` model-zoo table is enabled under separate `mmhuman3d:*` identities.** The adapter pins one public archive and retains the same-row source configuration, not the broad list of supported human-model methods. | Adds HMR, SPIN, VIBE, HybrIK, PARE, ExPose, PyMAF-X, and related 3D human-model release evidence with exact source documentation, reference-only weights, and linked Python training/evaluation configurations. | This source covers only explicit release-table rows. Dataset/body-model downloads, prose-only methods, and any absent/retired checkpoint are not inferred; no source code or model bytes are executed or downloaded. |
| [MMAction legacy model zoo](https://github.com/open-mmlab/mmaction/blob/master/MODEL_ZOO.md) | **Every direct recognized checkpoint URL in the retained project's exact root `MODEL_ZOO.md` tables is enabled under independent `mmaction-legacy:*` identities.** The source is pinned through one bounded archive read and excludes all other repository paths. | Adds historical action-recognition, action-detection, and spatial-temporal action-detection releases with table/heading provenance, direct reference-only checkpoints, and the original source repository/documentation. | This is the original MMAction document, not MMAction2 or an assertion that every described architecture has a published artifact. It neither follows historical links nor transfers source, model, or dataset bytes. |
| [MMFashion Model Zoo](https://github.com/open-mmlab/mmfashion/blob/master/docs/MODEL_ZOO.md) | **Every direct-artifact row in the historical first-party tables headed `Backbone` or `Model type` is enabled at a pinned commit.** It rejects other document tables instead of treating performance text as a model assertion. | Adds attribute/category prediction, clothes retrieval, landmark detection, compatibility, segmentation, and virtual try-on evidence with first-party row/heading provenance, model/backbone labels, and direct Google/Baidu/PyTorch artifact references. | This is an archival task-model plane, not proof that each artifact remains reachable or every MMFashion component was released. Hosts and artifacts are references only; the parser neither follows them nor downloads bytes. |
| [OpenUnReID Model Zoo](https://github.com/open-mmlab/OpenUnReID/blob/master/docs/MODEL_ZOO.md) | **Every visible direct-artifact row in a first-party `Method`/Download table is enabled at a pinned commit.** HTML-commented examples are explicitly excluded before parsing, and relative method links are retained only as their source-declared label. | Adds unsupervised/domain-adaptive person and object re-identification methods with task/table provenance, exact method labels, first-party documentation/repository links, and direct reference-only Google Drive artifacts. | The source is a historical public table, not a proof that a Drive artifact remains reachable or all configured methods were released. It neither follows method/config links nor transfers any model bytes. |
| [Kaggle Models](https://github.com/Kaggle/kaggle-cli/blob/main/docs/models.md) | **Public `models list` API enabled with historical versions.** Follow opaque page tokens for models and variations in `createTime` order and retain every exposed version. Per-version file metadata is available with `include_version_files = true`. | Owner/model identity, framework variation, version-scoped metadata, card/provenance/training-data links, declared license, external base-model lineage, and reference-only download URLs. Also reaches models distributed through the TensorFlow/Kaggle ecosystem. | Honor provider pagination and rate responses. File metadata is opt-in because it adds requests for each version; the public listing's reported total does not include private or unavailable models. |
| [CivitAI Models](https://github.com/civitai/civitai-developer-docs/blob/main/site/reference/models.md) | **Public `models` API enabled.** Follow the provider's opaque newest-first cursor over every returned model; retain public metadata even when a version is archived, unavailable, or has no downloadable file. | CivitAI model, model-version, and model-file IDs; SHA-256 when declared; model/version cards; provider-declared base-model-family lineage; free-text paper/code/card URLs; and reference-only file download URLs. This adds a large independent generative-image and derivative-model plane. | Honor pagination, response limits, and terms. The public API can region-filter NSFW results even with `nsfw=true`; each run therefore records endpoint-visible coverage, not a claim to private, gated, or region-withheld inventory. |
| [Zenodo Model records](https://zenodo.org/api/records?q=resource_type.type:model&sort=oldest&size=100&page=1) | **Exact public `resource_type.type:model` query enabled.** Start at the oldest record, require Zenodo's reported total to stay stable through a scan, and follow only a next URL that preserves the exact query, sort, page size, path, and origin. | Adds exact Zenodo record and record-DOI identities, the source-declared concept/version/DOI pages, titles/descriptions, license/resource-type metadata, explicit related publication/software/dataset links, and direct file references with raw checksum metadata. | Zenodo's `Model` resource type is cross-domain—e.g. it can mean an ontology or a 3D object—so this source makes no model declaration by itself. The normal conservative extractor can emit neural-model candidates only when the record's own text supports neural scope. It is a current public listing with no deletion semantics; files are never downloaded. |
| [OpenCSG Hub](https://github.com/OpenCSGs/csghub-server) | **Public `models` API enabled.** Traverse every reported `page`/`per` record in `recently_update` order; abort and restart if the provider-reported total changes during a scan rather than silently treating a moving inventory as a complete snapshot. | OpenCSG model path and repository ID; model page and clone reference; description/readme URLs; tags/license metadata; and exact source-declared Hugging Face or ModelScope mirror paths. The clone URL is a non-crawled reference, never an instruction to transfer repository or LFS bytes. | Honor response rate limits and terms. The public endpoint reports its visible inventory, not private/gated content. Pagination is count-stable rather than cursor-snapshotted, so a completed run is endpoint evidence at its observation time, not an immutable provider export. |
| [ModelScope OpenAPI](https://github.com/modelscope/modelscope_hub/blob/main/src/modelscope_hub/_openapi.py) | **Bounded public discovery enabled.** The provider documents four list orders—`default`, `downloads`, `likes`, and `last_modified`—and caps anonymous pagination at 3,000 rows per sort. The adapter traverses each configured window exactly, with no search term or model-name filter. | Adds exact `modelscope:model` IDs, first-party model-card and detail endpoints, descriptions, task/tag metadata, dates, and any direct URL declared in a list description. It deliberately does not fetch a card, repository, paper, or artifact for each result. | A completed scan means the configured windows are complete, not that all provider-reported models have been seen. The observed-row count can include the same source model under more than one documented ordering; source-native IDs reconcile those observations. |
| [NVIDIA NGC catalog](https://docs.ngc.nvidia.com/sdk/api.html) | **Guest-visible public `MODEL` group enabled.** Query the first-party catalog with a deterministic name/resource-ID order, use its total to validate each page, and abort/restart if that total changes during the sweep. | Adds exact `ngc:model` IDs, card URLs, descriptions/labels, and NGC's declared latest-version ID as a release handle. A later exact-version metadata resolver can read one card's source-declared Markdown links and release data, preserving paper/code/documentation URLs without deriving a download URL. | The catalog is a mutable, paginated provider view, not a versioned historical export. A complete sweep never authoritatively deletes earlier records, and it excludes private, gated, removed, or unlisted content. Card resolution is bounded per entry; historical versions, archives, and weight bytes remain outside this source. |
| [NVIDIA NeMo checkpoint tables](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/results.html) | **First-party ASR, TTS, audio, diarization, and self-supervised-speech tables enabled.** The adapter admits only table rows whose first column is a model field and whose same row declares an exact Hugging Face or NGC card; it retains the row and its model-scoped documentation/card links. | Adds NeMo's source-declared released checkpoint names and direct cards. Exact Hugging Face repository IDs and unambiguous per-model NGC IDs bridge existing provider entries; a shared NGC collection card deliberately remains a resource rather than a false identity merge. | This is current documentation coverage, not a complete historical NeMo export. It does not call package model-list APIs, retrieve model cards, infer weight URLs, or turn citation keys into paper links. |
| [Ollama library](https://ollama.com/library) | **Every model-family card structurally listed in the returned first-party library document is enabled.** The extractor takes each card's declared library slug and title attribute, not surrounding marketing text, and treats the document as one conditional-GET snapshot. | Adds exact `ollama:model-family` identifiers and direct model-card references for Ollama-distributed families. The provider card remains distinct from an upstream model, source repository, or original weight host unless another source states a relationship. | The source is document-level coverage only: it does not establish that the library is a historical census or that a card’s listed tags are a complete artifact-version inventory. No manifest, model page, blob layer, or weight is fetched. |
| [Cloudflare Workers AI catalog](https://developers.cloudflare.com/workers-ai/models/) | **Every card structurally present in the returned first-party catalog document is enabled.** The extractor requires the card's declared `@cf/...` runtime ID, display label, and canonical documentation path to occur together. | Adds exact `cloudflare-workers-ai:model` IDs and first-party detail-page references for models currently hosted or served by Workers AI. This is an independent availability observation; it does not merge the runtime ID with an origin model, repository, paper, or weight artifact. | This conditional-GET document snapshot is a current provider catalog, not a historical census or a complete version/artifact inventory. No card detail page, API endpoint, manifest, or model bytes are fetched. |
| [OpenRouter Models API](https://openrouter.ai/docs/api/api-reference/models/list-all-models-and-their-properties) | **All-output-modality public catalog enabled.** Request `output_modalities=all`, omit pagination parameters, require its reported `total_count` to equal the received row count, and fail if a next page is present. | Adds every currently router-available model ID, provider model page, declared modalities/capabilities, description URLs, endpoint locator, exact router-alias edge, and source-declared Hugging Face repository relation. It covers hosted commercial and open models without claiming that a routed model is an OpenRouter-owned artifact or that weights are downloadable. | Honor current terms and 429/retry headers. This is a current availability snapshot: private, removed, never-routed, or historical models are not covered; a missing row only says it is absent from that observed catalog. |
| [Replicate Models API](https://replicate.com/docs/reference/http) | **Token-gated public catalog implemented.** When `REPLICATE_API_TOKEN` is explicitly present, follow the source-supplied same-origin cursor URLs from its List Models endpoint; no token is stored in records, state, or URLs. | Adds model cards and exact latest-version IDs, plus source-declared README/description, code, paper, weights, and license URLs. It then follows each listed public model’s cursor-paginated version history, one page per checkpointed fetch; version metadata is retained without downloading outputs or weights. | Replicate documents that List Models contains only public models meeting its displayworthy criteria, so this is not a public-model census. The source is inactive without an explicit token and must never enumerate private account state. |
| [OpenAI Models API](https://developers.openai.com/api/reference/resources/models) | **Credential-gated public-owner catalog implemented.** When `OPENAI_API_KEY` is explicitly present, retain only rows whose documented `owned_by` value is in the configured public-owner policy (`openai`, `system` by default); other rows are counted but never stored. | Adds exact `openai:model` IDs, owner, API creation/shutdown metadata, an exact provider-resource endpoint, and the official provider documentation link. These are current availability observations, not source-code, paper, checkpoint, or weight assertions. | The API is account-scoped, so this source must never turn private fine-tunes or organization models into registry records. It is inactive without a key, never records the key, and does not treat an absence from one account response as deletion or historical nonexistence. |
| [OpenAI public all-models documentation](https://developers.openai.com/api/docs/models/all) | **Public catalog enabled.** Structurally retain each official detail-page destination, including cards labelled deprecated; the final URL path becomes the exact `openai:model` ID. A bounded per-card resolver preserves declared snapshot IDs as releases and merges with the optional API source only by that exact ID. | Adds first-party model-documentation resources and documented snapshots for public current and deprecated entries without an API key or account-visible list. It never treats page navigation as a paper/code/weight relation. | This is a mutable documentation snapshot, not proof that every historically served or undocumented model is listed. It deliberately excludes the catalog/compare index paths and does not query accounts, fine-tunes, or organization models. |
| [Anthropic Models API](https://platform.claude.com/docs/en/api/models/list) | **Credential-gated provider catalog implemented.** When `ANTHROPIC_API_KEY` is explicitly present, honor the documented `anthropic-version` header and `has_more`/`last_id` pagination contract before retaining every API-available model row. | Adds exact `anthropic:model` IDs, display/release metadata, capability metadata retained in raw source evidence, and each direct API model-resource link. The model response is provider availability evidence, not a declaration of papers, code, weights, or model ownership outside Anthropic's API. | The source is inactive without an explicit key, never retains the key, and treats the account's API-visible list as a mutable availability view rather than a global historical or artifact catalog. Provider retirement and history must be supplied by separate source-backed records. |
| [Google Gemini Models API](https://ai.google.dev/api/models) | **Credential-gated provider catalog implemented.** When `GEMINI_API_KEY` is explicitly present, honor the documented `pageSize`/`pageToken` pagination contract, retaining every returned model resource name exactly. | Adds exact `google:gemini-api-model` resource-name identifiers, display/version/capability metadata, declared base-model relations, and each direct API model-resource reference. It never claims a serving name is a paper, code repository, checkpoint, or downloadable weight. | The key is supplied only as the request's `key` parameter and is never stored in a source URL, checkpoint, raw evidence, or link. This is a mutable API-availability view, not a historical model census; each provider resource link remains non-crawling because it requires authentication. |
| [Groq Models API](https://console.groq.com/docs/api-reference) | **Credential-gated current provider catalog implemented.** When `GROQ_API_KEY` is explicitly present, retain every row from the endpoint Groq documents as its active model list. | Adds exact `groq:model` serving IDs, source-declared owner, creation timestamp and capability metadata, plus direct API resource references. A hosted row remains availability evidence; it does not assert that Groq owns the model or provides its paper, code, or weights. | The API key is used only in the request header and is never stored. The documented list is current and active-only, so absence is not historical deletion evidence; per-model authenticated endpoints are retained as non-crawling references. |
| [Mistral public model documentation](https://docs.mistral.ai/models) | **Public catalog plus checkpointed detail resolver enabled.** Structurally retain each first-party `/models/...` detail-page link from the published current, deprecated, and retired sections, then resolve the frozen worklist in bounded batches. | Adds exact `mistral:model-documentation` path identities, labels, and model-scoped documentation resources plus only the explicitly selected arXiv paper and exact `mistralai/*` Hugging Face card links. Navigation, social links, and a generic organization repository are excluded rather than guessed to be model resources. | The documentation catalog is not the account API: it deliberately avoids private fine-tunes. It is a bounded current-and-retired documentation plane, not proof of models Mistral never documented or later removed from this page. |
| [Cohere public model documentation](https://docs.cohere.com/docs/models) | **Public catalog enabled.** Structurally retain exact first-column API model names from the overview's `Model Name`/description tables, including entries marked live, deprecated, or retired. | Adds exact `cohere:model-documentation` identifiers and model-scoped provenance on the first-party overview page. Any paper, code, checkpoint, or deployment link requires separate source-declared evidence; this page alone does not infer it. | The public overview is intentionally used instead of Cohere's authenticated `List Models` API because the latter reports a `finetuned` field and can expose account-visible custom models. The overview is not proof of undocumented or removed models. |
| [Amazon Bedrock model cards](https://docs.aws.amazon.com/bedrock/latest/userguide/model-cards.html) | **Public catalog enabled.** Structurally retain every first-party `model-card-*.html` destination published by the at-a-glance catalog. | Adds exact `aws:bedrock-model-card` path identities, card resources, and documentation coverage for current Bedrock-hosted offerings. A later bounded card fetch can retain its source-declared serving IDs, launch/EOL metadata, service card, license, and other provider links. | This deliberately avoids the credentialed Bedrock discovery API because it can include account-specific custom or provisioned models. It is a current host catalog, not an origin-model identity assertion or a historical census. |
| [SageMaker JumpStart available models](https://docs.aws.amazon.com/sagemaker/latest/dg/jumpstart-foundation-models-latest.html) | **Public catalog enabled.** Structurally retain all rows from tables headed `Provider / Model Name / Model ID / Task` or `Model Name / Model ID / Task`, keeping the display label and exact deployment ID separate. | Adds `aws:sagemaker-jumpstart-model` deployment identities, provider labels, task and license context from a dated official inventory spanning open-weight, proprietary, and built-in offerings. The catalog page is retained as evidence for each model. | The document itself is a dated, mutable provider snapshot, not a history of removed models or proof that a hosted deployment is an origin paper, repository, or weight. It deliberately avoids Hub, Marketplace, and account-specific APIs. |
| [BioImage.IO](https://bioimage.io/) | **Official public exact-version collection enabled.** The adapter enumerates every upstream `type=model` resource, retrieves each versioned manifest and RDF/YAML card, and verifies the index-declared SHA-256 before publication. | Resource/version ID, model identity, code/config/documentation links, weight formats and checksums, citations, timestamps, and removal evidence. High-precision biological imaging coverage. | Each resource declares its own license; validation does not grant broader rights. Weight binaries are referenced rather than downloaded. |
| [MONAI Model Zoo](https://github.com/Project-MONAI/model-zoo) | **Official versioned bundle registry enabled.** The adapter resolves `Project-MONAI/model-zoo` at one public commit and enumerates every entry in `models/model_info.json`; a later revision emits tombstones for bundle entries no longer present. | Adds exact MONAI bundle model/release identities, source-repository revision, archive URL, and any source-declared SHA-1 without extracting the archive. It complements BioImage.IO rather than treating the two registries as interchangeable. | Registry source code follows its repository license. Each linked bundle archive and any embedded weight/data license must be assessed separately; unknown archive rights are never inferred from the registry. |
| [arXiv Complete Corpus](https://huggingface.co/datasets/secemp9/arxiv-complete) | **Public historical metadata snapshot enabled.** The adapter resumes in 10,000-record slices of bounded Parquet row groups using validated HTTP byte ranges, selecting only metadata rather than the corpus's 16 TB of paper content. | Backfills 3,148,796 paper identities, abstracts, categories, DOI/license links, and submission dates through the snapshot's 2026-08-27 boundary. It shares the live arXiv artifact namespace but never infers deletions from an older static snapshot. | The organizational/metadata layer is CC0; each paper's recorded license is preserved. Do not redistribute paper content merely because this source is public. |

The ONNX Model Zoo is an archival source, not a primary live distribution channel. Its
per-entry evidence is preserved with the complete index revision; old download URLs are
not relied on as available weight artifacts.

### ModelScope bounded public windows

ModelScope's public OpenAPI model endpoint is enabled as a deliberately bounded source,
not as a provider-complete snapshot. Its official client documents the constraint
`page_number * page_size <= 3,000`; the adapter exhausts each configured public sort window
and records the provider's reported total separately. It never marks that window union as
an authoritative snapshot, so it cannot tombstone model records that lie outside the
provider-exposed range. A later official full export, stable cursor API, or reconcilable
provider sharding contract could extend the source to complete coverage without changing
the identities already retained. The constraint is documented in the official
[ModelScope Hub client](https://github.com/modelscope/modelscope_hub).

## Priority 2 roadmap: publications, software, and cross-links

These sources enlarge recall and connect papers, code, data, releases, and citations.
Linked GitHub repositories are enriched by the frontier. The enabled GH Archive activity
plane discovers repositories from every public event in each processed hour, while the
first-party GitHub public-repository catalog advances a durable numeric-ID cursor and the
enabled Software Heritage origin plane supplies a much broader fixed historical inventory.
Neither archive nor the mutable public API is proof that every repository ever created is
present or still available.

| Source | Enumeration and identifiers | License and operational constraint |
| --- | --- | --- |
| [Crossref REST](https://www.crossref.org/documentation/retrieve-metadata/rest-api/) | Cursor through works and changes; DOI is primary. Publisher-deposited links and relations help verify papers. | Most bibliographic metadata is reusable, but abstracts and other deposited content may be copyrighted. Use the [polite pool and current rate headers](https://www.crossref.org/documentation/retrieve-metadata/rest-api/tips-for-using-the-crossref-rest-api/) with `mailto`; use bulk products for very large backfills. |
| [DataCite](https://support.datacite.org/docs/harvesting-datacite-doi-metadata) | REST, OAI-PMH, or public data file; query software resource types and follow `relatedIdentifiers`. DOI, version, SPDX/right fields and links join software, papers, and datasets. | Public metadata is [CC0](https://support.datacite.org/docs/datacite-data-file-use-policy). Use the public file for a complete backfill and API pagination for changes. |
| [OpenAIRE Research Graph](https://graph.openaire.eu/docs/apis/graph-api/research-products/) | API or [full JSONL graph](https://graph.openaire.eu/docs/downloads/full-graph); research products include publications, datasets and software with repository URLs and PIDs. | Graph dump is CC BY 4.0. Attribute OpenAIRE and retain original IDs; use the dump rather than exhausting search endpoints. |
| [Zenodo](https://developers.zenodo.org/) | **Enabled.** The records API covers declared Model resources; a separate full-repository OAI-PMH DCAT scan admits candidate neural-model evidence only when metadata and an explicit checkpoint distribution agree. Concept DOI groups version DOIs and related identifiers connect code/data/papers. | Metadata is CC0; files retain their deposited license. OAI resumption tokens expire quickly, so uninterrupted harvest runs must renew them promptly; the adapter stores metadata and links without downloading files. |
| [arXiv](https://info.arxiv.org/help/api/) | API/OAI for metadata and the official bulk route for scale. Preserve base arXiv ID and every version. Subject categories cover CS, statistics, physics and quantitative biology. | Full text follows each submission's license. Follow the API's request-delay policy; do not scrape article HTML or PDFs as an enumeration method. |
| [HAL](https://api.hal.science/docs/oai) | **Enabled.** First-party OAI-PMH enumerates HAL's entire research-output metadata stream with no topic, institution, venue, author, or model filter. The adapter begins at HAL's declared 2002-09-23 lower boundary, freezes date-granularity windows, and follows opaque resumption tokens while preserving HAL IDs, versions, DOI, creators, abstract, rights, sets, and landing/full-text links. | HAL exposes metadata harvesting without registration; its OAI terms prohibit commercial reuse of harvested metadata. The registry retains metadata and links, never article files, and preserves source-declared rights rather than inferring them. |
| [PLOS Search API](https://api.plos.org/solr/faq) | **Enabled.** The publisher-owned Solr API scans all full PLOS documents in publication-date/DOI order, without a journal, topic, author, or model filter. The offset adapter freezes a date window and restarts it if the declared total moves, retaining DOI, metadata, abstract, article type, rights, and non-crawling publisher/JATS references. | PLOS asks high-volume users to use its designated TDM bulk route; this adapter is bounded metadata discovery and does not retrieve article files or scrape HTML. Article rights remain the source-declared terms even where PLOS enables broad text mining. |
| [OpenReview](https://docs.openreview.net/getting-started/using-the-api) | **Enabled.** Globally traverses public notes from both API v1 and v2 without venue, invitation, subject, or model filters; preserves revisions and workflow-note evidence while labeling only publication-shaped notes as papers. OpenReview note ID is primary. | [Terms](https://openreview.net/legal/terms) place metadata under CC0, while article rights follow the article's license. Private/confidential notes are necessarily outside the public corpus. |
| [GitHub public repositories](https://docs.github.com/en/rest/repos/repos#list-public-repositories) + [GH Archive](https://www.gharchive.org/) | **Enabled.** The first-party `GET /repositories` source traverses every public repository returned after its durable numeric `since` cursor, with no repository-name, owner, language, topic, activity, or model-term filter. Every row keeps GitHub's numeric ID and enters the same metadata/README frontier. GH Archive independently retains every public event in each closed hour and projects its repository IDs with event evidence. | GitHub's mutable API does not expose a deletion feed and full historical traversal is rate-bound, so the source reports only the reached public-ID range; private, deleted, and not-yet-reached repositories remain outside the observed catalog. REST calls are quota-bound; use an optional least-privilege `GITHUB_TOKEN` for public metadata at scale. The [license endpoint](https://docs.github.com/en/rest/licenses/licenses) recognizes only certain root licenses; missing or unknown never means open. Use the API, not HTML scraping, under GitHub's [acceptable-use policy](https://docs.github.com/en/site-policy/acceptable-use-policies/github-acceptable-use-policies). |
| [Software Heritage full-graph export](https://docs.softwareheritage.org/devel/swh-export/graph/dataset.html) | **Enabled historical origin plane.** The adapter dynamically approves the latest settled full-graph release using the official release document plus `meta/export.json`, freezes the entire ORC `origin` inventory, and lands every binary URL row in Parquet. Each committed shard is immediately projected without model terms; structurally exact GitHub origins enter the ordinary repository frontier. | The [origin schema](https://docs.softwareheritage.org/devel/swh-export/graph/schema.html) supplies URLs, not repository contents or a neural-model assertion. Model admission still needs reachable GitHub metadata/README evidence. This is a snapshot of [Software Heritage's holdings and known limitations](https://docs.softwareheritage.org/user/using_data/index.html#possible-bias-and-limitations), not a complete census of all software. Retain release/object/local digests and follow the [bulk-access terms](https://www.softwareheritage.org/legal/bulk-access-terms-of-use/). |
| [Wikidata](https://www.wikidata.org/wiki/Help:Data_access) | SPARQL, dumps, and entity JSON/RDF provide QIDs, aliases, DOI/arXiv/repository links and structured relationships. | Structured data is [CC0](https://www.wikidata.org/wiki/Wikidata:Licensing). It is an alias/ontology seed, not a comprehensive current model catalog. |

The [Papers with Code data repository](https://github.com/paperswithcode/paperswithcode-data)
provides a CC BY-SA historical paper-code, method, dataset, and evaluation graph. It is
valuable as corroborating historical alias and relation evidence, but it is a retired
snapshot and cannot be assumed complete or current. Its share-alike attribution
requirements must remain visible in derivative exports.

[alphaXiv's official MCP service](https://www.alphaxiv.org/docs/mcp) can enrich an
already identified arXiv paper with extracted text and codebase exploration. Its broad
paper-discovery tool is ranked, agentic search rather than an exhaustible corpus export,
so it is not MODELOME's enumeration backbone or a completeness signal. The implemented
bounded job selects only exact arXiv IDs retained on active artifacts from other sources,
preserves alphaXiv output as source-scoped evidence, deprioritizes failed IDs so they
cannot starve newly discovered work, and can explicitly refresh old successful evidence.
`modelome daily` runs it when `ALPHAXIV_API_KEY` is configured; `modelome enrich-alphaxiv` runs it
directly. Repository-link claims remain evidence to verify against the paper or code host.

### Institutionally accessible publisher text

Publisher full text can be decisive for a model's training/evaluation details, inputs and
outputs, ablations, and links to code or weights. It is not needed merely to construct the
citation graph: Crossref, OpenAlex, PubMed, Europe PMC, and source-declared links remain the
normal bibliographic evidence plane. Nor does a Berkeley login automatically authorize an
unattended crawler or text-and-data mining; the applicable institutional license, publisher
terms, robots policy, and any designated TDM route control that use.

For a single article whose permitted use has been checked, an authorized user may manually
obtain and extract the article into UTF-8 text, then run:

```bash
modelome ingest-institutional-text \
  --doi 10.0000/example \
  --title "Example model paper" \
  --text-file /absolute/path/to/extracted-paper.txt \
  --landing-url https://publisher.example/article \
  --access-confirmed
```

The command requires an exact DOI and explicit confirmation. It retains the text in the private
evidence store under `institutional-restricted`, invokes the ordinary model/code-link extractor,
and records the landing page only as a non-crawling locator. It never reads a browser session,
copies cookies, downloads a PDF, or activates on a schedule. The public metadata exporter omits
artifact-revision text and raw evidence, so restricted publisher text cannot enter a distributed
bundle through this path. Do not use this workflow where a license requires destruction after
analysis or otherwise prohibits local retention; add a non-retaining per-publisher workflow only
after the exact contractual requirements are known.

## Priority 3 roadmap: scientific-domain repositories and extraction

Scientific neural models often lack familiar ML tags, use domain-specific names, or
appear only in a paper and repository. Domain ingestion therefore enumerates complete
corpora and applies neural-model/relation extraction after acquisition. A molecular
diffusion model can be in scope; a physical diffusion equation or non-neural Monte Carlo
method is not automatically a model entity.

### Biology and biomedicine

- The Priority 0 bioRxiv, medRxiv, PubMed, PMC, and Europe PMC adapters provide the
  unfiltered literature plane. BioImage.IO and the MONAI Model Zoo, listed above, supply
  complementary structured packaged neural-model release planes for biological imaging.
- Paper/code extraction remains necessary for molecular, protein, genomics, clinical,
  and non-imaging neural systems. Domain databases and repositories can supply useful
  cross-links, but their simulations, statistical models, and curated biological models
  remain contextual unless retained evidence establishes a substantive neural component.

### Physics

- [INSPIRE REST](https://github.com/inspirehep/rest-api-doc) covers high-energy physics
  with record IDs, DOI, arXiv and ORCID links. Its published service limit is 15 requests
  per five seconds, with result-window constraints; use cursor/date partitions and obey
  its [terms](https://help.inspirehep.net/knowledge-base/terms-of-use/).
- [NASA ADS](https://ui.adsabs.harvard.edu/help/api/) covers astronomy and astrophysics
  with stable bibcodes, DOI and arXiv links. Authentication is required and the current
  [rate-limit policy](https://ui.adsabs.harvard.edu/help/policies/rate-limits) gives the
  standard user token a daily request budget.
- arXiv, OpenAlex, Crossref, OpenAIRE, Zenodo and linked code fill broader physics areas.

### Chemistry and materials

- Enumerate OpenAlex, Crossref and DataCite chemistry/materials records; follow OpenAIRE,
  Zenodo and [Figshare API](https://docs.figshare.com/v2/) software/data relations into
  code and model releases.
- Europe PMC contributes chemical and biochemical literature and annotations.
- Do not depend on undocumented ChemRxiv internal endpoints. Discover ChemRxiv records
  through deposited DOI metadata and scholarly indexes until it publishes a supported
  machine API suitable for bulk use.
- Extract molecular graph, conformer, protein, reaction, materials and field/simulation
  diffusion relations semantically. The literal word `diffusion` is neither necessary
  nor sufficient to classify a model.

## Available generic adapter and Priority 4 roadmap: provider catalogs

Provider model-list APIs describe availability, not necessarily the originating model
or an immutable release. OpenRouter's complete public catalog is enabled above as a
current provider-availability plane. The implemented `json_catalog` adapter can map other
documented JSON endpoints without provider-specific Python code; see
`config/provider-sources.example.toml`. It supports cursor or same-origin URL pagination,
header or query-parameter credentials, bounded response bodies, model-card URLs, timestamps,
`base_model` relations, and opt-in release identity. Replicate, OpenAI, Anthropic, and Gemini
and Groq activate only with their explicit API credentials; other provider rows below remain candidates.

Today those entries become provider-page artifacts linked to model identities. The
schema does not yet have a deployment entity, so provider IDs should be treated as
mutable availability observations unless the operator has evidence otherwise. Set
`provider_id_is_release = true` only for IDs the provider documents as immutable.

| Provider | Primary machine endpoint | Identity and caveat |
| --- | --- | --- |
| OpenRouter | [Public models list](https://openrouter.ai/docs/api/api-reference/models/list-all-models-and-their-properties) | **Enabled.** Catalog model ID is routing-availability evidence. Preserve source-declared aliases and Hugging Face links as relations, not cross-provider identity merges. |
| Replicate | [List public models](https://replicate.com/docs/reference/http) | **Token-gated.** Cursor through the public, displayworthy model list and preserve model/version IDs and explicit resource URLs. No token means no source activation. |
| OpenAI | [`GET /v1/models`](https://developers.openai.com/api/reference/resources/models) | **Credential-gated public-owner source enabled.** Retain provider IDs, owner, creation/shutdown metadata, and current availability only when `owned_by` matches the configured public-owner policy. Never retain an account-private or organization-owned row. |
| OpenAI public documentation | [All models](https://developers.openai.com/api/docs/models/all) | **Enabled.** Retain each current/deprecated direct model page under its exact `openai:model` URL ID; resolve one page at a time only to retain documented snapshots as releases. No account or organization list is queried. |
| Anthropic | [Models list](https://platform.claude.com/docs/en/api/models/list) | **Credential-gated source enabled.** Follow the documented `has_more`/`last_id` cursor contract with the required API-version header; retain provider IDs, display/release/capability metadata, and current API availability only. |
| Google Gemini | [`models.list` and `models.get`](https://ai.google.dev/api/models) | **Credential-gated source enabled.** Follow the documented `pageToken` cursor and retain exact model resource names, display/version/capability metadata, and declared base-model relations. Never persist its required query key or crawl credentialed item resources. |
| Groq | [Models API](https://console.groq.com/docs/api-reference) | **Credential-gated source enabled.** Retain every provider-documented active model row with exact serving ID, owner, creation timestamp, and current availability only; never persist the key or crawl credentialed item resources. |
| Mistral | [Public model documentation](https://docs.mistral.ai/models) | **Enabled.** Structurally retain every public model-detail link from the current and retired sections, then resolve its explicit arXiv and `mistralai/*` Hugging Face links in checkpointed batches. This avoids the account-scoped API model list, which can include fine-tunes; treat documentation paths as source-scoped model identities and pages, not a complete provider history. |
| Cohere | [Public model overview](https://docs.cohere.com/docs/models) | **Enabled.** Structurally retain exact model names from documented current/deprecated overview tables. This avoids the authenticated list, which can include fine-tunes; treat each name as public documentation evidence, not a complete provider history. |
| AWS Bedrock | [Public model-card catalog](https://docs.aws.amazon.com/bedrock/latest/userguide/model-cards.html) | **Enabled.** Retain each publicly listed first-party card as a source-scoped model identity and later bounded enrichment resource. This avoids account-scoped custom/provisioned model discovery and does not identify a Bedrock serving card with the model's origin paper, code, or weights. |
| AWS SageMaker JumpStart | [Available foundation models](https://docs.aws.amazon.com/sagemaker/latest/dg/jumpstart-foundation-models-latest.html) | **Enabled.** Retain every public table row's exact deployment model ID and display name without calling account-scoped hub or Marketplace APIs. This is dated catalog evidence, not origin-model or historical-census evidence. |
| Azure AI Foundry | [Models list](https://learn.microsoft.com/en-us/rest/api/aifoundry/accountmanagement/models/list) | Publisher/model/version and regional availability; distinguish catalog listing from a provisioned deployment. |
| NVIDIA NGC | [Catalog API/CLI](https://docs.ngc.nvidia.com/sdk/api.html) | **Enabled for the guest-visible current `MODEL` group.** Preserve NGC-native model and declared latest-version IDs plus model-card URL. The direct public version-metadata endpoint resolves one card's declared Markdown resources at a time. This is a mutable public listing, not a historical export; archives and artifact bytes remain outside scope. |

For a provider that offers only web pages, prefer an official sitemap, RSS/Atom feed,
OpenAPI document, JSON-LD, or documented export. Check robots and terms before retrieval.
Store factual metadata, URL, timestamp, digest, and short evidence locators rather than
republishing copyrighted page bodies.

## Priority 5: blogs and historical web discovery

[Common Crawl](https://commoncrawl.org/get-started) WARC/WAT/WET data and its
[CDXJ indexes](https://commoncrawl.org/cdxj-index) can discover blog-only models,
historical documentation, and pages that later disappeared. Use domain discovery and
semantic extraction, then seek an authoritative paper, repository, artifact, or archived
page as corroboration. The [Common Crawl terms](https://commoncrawl.org/terms-of-use)
make clear that underlying content retains the original owner's rights and terms; the
crawl is not a license-cleaning mechanism.

The collection/WET-manifest control adapter, bounded WET payload loader, and immutable
discovery projection are implemented. They enumerate every published collection and
shard without a domain filter, verify stable control identity and available object/WARC
digests, bound gzip and record expansion, retain every conversion document, and emit
separate Parquet tables for exact URL relations and evidence-backed neural-model
candidates. A restart-safe streaming merge now admits candidate-bearing documents and
exact GitHub relations into the searchable registry while leaving non-actionable
documents in the lossless projection. CDX/WAT hidden-link discovery, removal processing,
and a public-export rights projection remain.

Web search and crawl results are candidate evidence, not proof of model identity. Honor
robots, removals and site policy, and do not copy full page text into public exports.

## Future corpus-scale rollout and coverage gates

This rollout begins only after the entry-first pipeline described in
[Entry-first modelome](entry-first.md) has a stable `ingest-paper` contract, fixtures,
work-item state, and a validated minimum entry shape. The stages below describe how to
extend that proven per-paper path; they are not the current execution plan.

### Available foundation

Implemented: daily-capable Hugging Face discovery with monthly created-at windows, bounded ModelScope Model OpenAPI windows, guest-visible NVIDIA NGC Model catalog rows, first-party NVIDIA NeMo checkpoint tables, historical GluonCV Model Zoo rows across six vision task planes, TorchVision static weight-enum releases, TensorFlow's retained TF1/TF2 Detection, DeepLab, Slim, Model Garden vision/NLP, and TPU release tables, CenterNet, CenterTrack, AlphaPose, pycls, DINO, DINOv2, Barlow Twins, SwAV, I-JEPA, DiT, MAE, ESM, BEiT/BEiT v2/BEiT-3, Segment Anything, SAM 2, ImageBind, DETR, ConvNeXt, Swin Transformer, DeiT, CaiT, ResMLP, PatchConvNet, PySlowFast, TimeSformer, VISSL, MaskFormer, Mask2Former, legacy Detectron, detrex, Detic, and X-AnyLabeling model-zoo rows, PyTorch Hub research-model pages plus the curated-listing extension, Ultralytics model families, Ollama library cards, Cloudflare Workers AI catalog cards, public OpenAI, Mistral, and Cohere model documentation, Amazon Bedrock model cards plus its Region/lifecycle matrix, SageMaker JumpStart public catalog, Oracle OCI and Azure Foundry catalogs, Zenodo Model-resource records, OpenRouter Models, credential-gated Replicate and OpenAI provider APIs plus Anthropic, Google Gemini, Groq, xAI, Together, Cerebras, DeepSeek, and SambaNova provider models, Cerebras public model documentation and Perplexity public router-model documentation, Kaggle Models with historical versions, CivitAI, OpenCSG Hub, Papers With Code official paper/code links and paper-linked method candidates, the full ACL Anthology XML collection manifest, BioImage.IO,
MONAI Model Zoo, Scenic, StarDist, Coqui TTS, DGL-LifeSci including its generative checkpoint registry, Uni-MOF, OpenFold and OpenFold3 parameter registries, Argus, Google NeuralGCM and Pangu-Weather checkpoint registries, CompVis Stable Diffusion first-stage and latent-diffusion README checkpoints, MACE-OFF23 and MACE foundation weights, RFdiffusion pretrained checkpoints, Cellpose, PyG GPSE, nnU-Net v1, CompVis latent diffusion, Diffusion Policy, Chemprop CheMeleon, MindSpore Model Zoo, Google Robotics RT-1, AlphaFold parameter-archive, GraphGPS GPS-deep, Gensim pretrained-weight, OpenCV DNN, and legacy MediaPipe checkpoint sources, T5X and ProteinMPNN checkpoint registries, OpenML flow implementations, robomimic policies, LightGBM, imbalanced-learn, and River technique catalogs, SciSpaCy pretrained pipelines, PaddleX model listings, WeNet pretrained models, PaddleNLP Taskflow UIE checkpoints, Pelican and Galaxea G05 VLA policies, FAIR Chemistry OCP legacy models, Dopamine checkpoint bundles, Kaldi model declarations, and GPT4All's model catalog; xAI credential-gated image/video model-family APIs and model version/lifecycle metadata; public Mistral lifecycle documentation and public Gemini, OpenAI snapshot, and Groq deprecation documentation, Azure Foundry/Anthropic/AWS Bedrock lifecycle documentation, OpenAlex publication/update, live arXiv plus the user-provided public arXiv metadata snapshot, OpenReview,
Crossref, Europe PMC, DataCite, PMC, bioRxiv/medRxiv, global OSF community-preprint, and EarthArXiv streams; complete arXiv and PMC bootstrap;
structural scikit-learn, sktime, aeon, Darts, PyOD, PyGOD, Statsmodels, tslearn, AutoGluon Tabular, PyTorch Forecasting, GluonTS, NeuralForecast, StatsForecast, PyTorch Tabular, TorchGeo ML-components and its source-declared pretrained-weight registry, AllenNLP historical model archives, five Fairseq NLP model zoos (RoBERTa, BART, XLM-R, mBART, and transformer language models), Ollama family cards and tag releases, and Segmentation Models PyTorch ML-component, Torchvision family and source-declared release, Torchaudio pipeline source-declared release, Keras Applications, KerasHub family and source-declared preset, PyTorch Geometric, TorchDrug, DeepChem, Transformers,
PaddleNLP transformer-family and pretrained-embedding, PaddleClas source-declared inference releases, PaddleDetection, Paddle3D, PaddleGAN, PaddleSeg, PaddleVideo, PaddleSlim, Fairseq wav2vec/wav2vec 2.0/translation table releases, Omnilingual ASR literal model cards, Seamless multilingual model-table releases, SONAR text-model releases, and AudioCraft MusicGen/MusicGen-Style/AudioGen/MAGNeT/JASCO plus EnCodec and AudioSeal first-party model-card links, PaddleRec recommendation-technique rows, PaddleOCR V2 and current model-table releases, MMEngine OpenMMLab/MMClassification runtime checkpoint maps, OpenAI CLIP/Whisper literal checkpoint maps and RL Clarity Procgen/IMPALA `.jd` pretrained-policy checkpoints, NLTK model-data archives, exact ref-pinned PyTorch Hub loader calls, OpenMMLab current/archival model-index including final MMClassification 0.x, MMHuman3D, original MMAction, MMFashion, and OpenUnReID table releases, Sentence Transformers pretrained-model, spaCy pipeline, Stanza resources, and Diffusers catalogs;
Semantic Scholar, PubMed, OpenAIRE, Common Crawl, GH Archive, and Software Heritage bulk
controls and bounded payload loading;
atomic Parquet evidence commits plus immutable sharded Parquet landing; exact-ID
model/release resolution; historical window backfills; linked code/page enrichment; a
Semantic Scholar model-candidate projection; exact-ID alphaXiv enrichment; a
provider-neutral JSON catalog adapter; acceptance-manifest reporting; and read-only
Epoch evaluation that never contaminates discovery.

Still required before calling the foundation broad: a unified corpus-scale searchable
projection across every landed row; package/weight expansion and configured provider
enumeration; more first-party code/weight/provider sources; normalized rights data;
source-specific complete-inventory reconciliation; and independently maintained
coverage and extraction-quality runs. The bundled cross-family and cross-science fixture
is an acceptance sentinel, not a full-family census.

### Later Priority 0: global scholarly and biomedical publication plane

1. Feed the implemented Semantic Scholar paper-state and evidence-bearing candidate
   projections, including every ordered update/delete diff, into the unified cross-source
   entity/search projection without weakening exact-ID resolution.
2. Execute complete historical `details` and `pubs` backfills for both bioRxiv and
   medRxiv plus bounded timestamp-window backfills for global OSF preprints and
   EarthArXiv; add monthly JATS/TDM reconciliation, and keep the implemented daily
   streams running with their independent checkpoints.
3. Exhaust the implemented PubMed baseline/delta and PMC OAI-PMH loaders, then add PMC
   AWS and Europe PMC bulk-inventory reconciliation as independent completeness checks.
4. Pass independently sampled global-paper and bio/medicine recall and
   scope-classification suites across preprints, journal articles, JATS full text, code
   links, and model repositories. The suites must include neural models, neural hybrids,
   and non-neural hard negatives. Keep *Attention Is All You Need* (`arXiv:1706.03762`)
   as an explicit paper-identity sentinel: it must be found through independent arXiv
   and Semantic Scholar enumeration even when another aggregator has incorrect dates or
   identifiers. The sentinel is evaluation data and must never become a discovery seed.

### Next phase: architecture breadth and lineage

1. Extend the enabled Zenodo model-resource source and add further first-party model registries; supplement the implemented GitHub activity and Software
   Heritage origin planes with forge-native repository enumeration and preserved-content
   resolution; keep the implemented Crossref, DataCite, OpenAIRE, and arXiv streams running.
2. Extend the implemented evidence-backed paper/code/card/weights relation projection
   with explicit architecture/model/release lineage and bulk-corpus inputs.
3. Pass sampled GAN, transformer and general-diffusion recall suites spanning different
   years, venues, libraries and hosting sites. The suites test discovery; they do not
   become query terms.

### Scientific-domain phase: chemistry, physics, and biology

1. Add INSPIRE, ADS and Figshare while retaining BioImage.IO, the Priority 0 biomedical
   plane and broad OpenAlex/arXiv/Crossref coverage.
2. Train and evaluate domain-aware model/relation extraction on independently sampled
   scientific literature and code.
3. Pass separate neural-model recall suites for chemistry/materials,
   physics/simulation, biology/medicine, and bioimaging, with non-neural hard negatives;
   report each domain rather than one blended score.

### Deployment and long-tail history phase

1. Snapshot provider model APIs and connect deployments to releases only with evidence.
2. Add compliant provider feeds and extend the searchable Common Crawl handoff with
   WAT/CDX hidden-link evidence, corroboration, and removal handling.
3. Run historical backfills, dead-link resolution, license reconciliation, and
   cross-source capture-recapture estimates.

Each phase is additive. An adapter should be treated as production-ready only after it
is idempotent, checkpointed, monitored, accompanied by documented rights/access policy,
and covered by relevant fixtures for pagination, updates, deletion semantics, throttling,
malformed records, and credentials.

## Source onboarding contract

Before enabling a new source, document and test:

- official owner and primary machine-readable endpoint;
- enumeration mechanism and supported bulk route;
- stable record ID, revision/version semantics, and deletion behavior;
- cursor or watermark, overlap strategy, and reconciliation method;
- authentication, rate limits, contact/user-agent rules, robots, and terms;
- metadata, text, file, and model-weight licenses separately;
- attribution text and downstream share-alike obligations;
- field mapping, raw-payload hashing, evidence locators, and fixture provenance;
- expected domains/artifact kinds for reporting, never retrieval filtering, and an
  independent coverage sample.

`config/sources.toml` contains source mechanics and field mappings only. Model names,
architecture families, and closed taxonomies do not belong in source configuration.
