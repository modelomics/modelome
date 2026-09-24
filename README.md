# Modelome

Modelome (MODELOME) is an evidence-backed registry for research entries: techniques,
models, architectures, releases, and the resources needed to understand or use them.
An entry connects the relevant paper(s), preprints, repositories, model cards, weights,
datasets, benchmarks, docs, and other evidence-backed links in one place. The current
implementation supplies the ingestion, provenance, identity, and bounded-link-enrichment
primitives for that registry.

## Install and build a dataset

Python 3.12 or newer is required. Install from this repository:

```bash
git clone https://github.com/modelomics/modelome.git
cd modelome
python -m pip install .
modelome --version
```

The installed `modelome` command and `python -m modelome` work from any directory.
The source catalog is bundled in the wheel; use `--sources-file` or
`MODELOME_SOURCES` to supply a custom catalog. Credentials are read from environment
variables; `.env.example` documents them and `.env` is not loaded automatically.

Run an offline example with a synthetic paper observation:

```bash
modelome --store data/example-store build-dataset \
  --input examples/papers.jsonl --output data/example-dataset
```

Build from real normalized paper observations using the same JSON/JSONL contract,
or export the current evidence store without network access:

```bash
modelome --store data/store build-dataset \
  --from-store --output data/exports/modelome-v1
```

To ingest a bounded source batch and build the cumulative dataset:

```bash
modelome --store data/store build-dataset \
  --source huggingface --max-pages 1 --output data/exports/modelome-batch-1
```

Repeat with the same store and a **new output directory** to resume its source
checkpoint. A partial source scan writes a bundle marked `partial` and exits with
code 1. Ingestion errors preserve completed checkpoints and publish no bundle.
These source batches are opt-in; the runner does not launch bulk loaders or a
frontier crawl. Each export includes all current admitted entries in the store.

The bundle contains `entries/entries.jsonl`, metadata Parquet tables under
`metadata/`, reusable `seeds.jsonl`, and `manifest.json` with the source commit,
counts, run outcomes, and SHA-256 checksums. Outputs are published atomically and
existing output paths are refused. A `complete` build means the requested work
finished; it does not claim exhaustive coverage of the modelome.

The same workflow is available as a Python library:

```python
from modelome.dataset import build_dataset

receipt = build_dataset("data/store", "data/exports/modelome-v2")
print(receipt.entry_count, receipt.source_commit)
```

See [Dataset builds](docs/dataset.md) for the library API, worklist schema, and
operating limits. For development, run `uv sync --locked --extra dev`,
`uv run pytest`, and `uv build`.

## Current operating phase

The immediate goal is **entry-first paper ingestion**, not a giant downloaded paper
dataset. We take one exact paper identity, collect its directly supported resources,
store their provenance and relationships, materialize a useful entry, and then move to
the next paper. Full-corpus ingestion will come later by feeding an enumerated paper
worklist through the same per-paper pipeline.

This is an important scope boundary:

- Do not launch a mass PDF/full-text download, archive-wide web crawl, or global weight
  download as part of the current work.
- Do build idempotent, checkpointed scripts that can ingest one paper correctly and can
  later be driven by every paper in an upstream source.
- Do store links and small, permitted evidence by default; heavyweight resources remain
  references unless an explicit later workflow needs their bytes.
- Do use loose, optional tags for field or department navigation. Tags are not a
  controlled ontology and never determine what can be ingested.

The full product contract, link semantics, minimal entry shape, paper-by-paper workflow,
and path to corpus scale live in [Entry-first modelome](docs/entry-first.md). Read that
document before treating the broad adapters described below as scheduled work.

“Universal” is a direction and a coverage contract, not a claim that the current
registry contains every model. No crawler can prove completeness over private,
deleted, unpublished, or undocumented work. MODELOME instead reports which configured
sources and date windows it processed, what remains unresolved, and recall against
an external acceptance corpus.

The entity boundary is narrower than the discovery boundary. MODELOME enumerates complete
source corpora without subject, journal, model-name, or method-keyword filters, then
retains the evidence needed to distinguish a neural model, a source-declared ML technique,
and surrounding context. Neural architectures, trained neural models, checkpoints, and
hybrids with a substantive neural component are central entries. A first-party framework
can also explicitly declare a classical estimator, transformer, or composition technique
as a documented technique entry; its source, tag, and status must make clear that it is
not a trained release. A mechanistic or statistical model, or a paper about physical
diffusion, does not qualify merely because its authors use the words “model” or
“diffusion.”

## Implemented building blocks

The following are available capabilities, not a directive to run every source or bulk
loader now. During the entry-first phase, use only the bounded source or link operations
needed to validate the per-paper workflow. Their later corpus-scale role is documented in
[Entry-first modelome](docs/entry-first.md#path-to-complete-paper-ingestion).
The default catalog currently has **398 enabled sources** (381 loadable without provider credentials); this is a configuration count,
not a claim that all upstream inventories have been exhausted.

The default catalog enables independent, unfiltered streams for:

- public Hugging Face model repositories, cards, current weight-file metadata, declared
  paper/code URLs, and `base_model` relations, with monthly created-at windows for bounded
  temporal discovery and optional historical revision-file traversal;
- the documented bounded union of ModelScope's public model-list orderings, retaining
  source-native model cards and metadata while explicitly respecting its 3,000-row
  per-sort pagination ceiling;
- every guest-visible current NVIDIA NGC `MODEL` catalog row, retaining its exact
  NGC model/version handles and first-party card URL while treating the moving,
  paginated listing as current provider evidence rather than a historical snapshot;
  a queued, exact-version metadata resolver then reads one card's declared Markdown
  links at a time without deriving download URLs;
- first-party NVIDIA NeMo ASR, TTS, audio, diarization, and self-supervised-speech
  checkpoint tables, retaining only rows with a direct Hugging Face or NGC model card
  and bridging to those provider identities only when the card URL is unambiguous;
- every model-family card structurally listed in Ollama's first-party library document,
  plus source-declared tag releases from each family's tags page, without fetching manifests or blobs;
- every current model card in Cloudflare Workers AI's first-party catalog, retaining its
  native runtime ID and documentation URL as provider-availability evidence only;
- every model currently declared by OpenRouter's complete public provider catalog,
  across all output modalities, including hosted commercial models, exact router aliases, and source-declared
  Hugging Face artifact links without implying that all entries have downloadable weights;
- every token-authorized, displayworthy public Replicate model card, retaining its declared
  code, paper, license, weight, README, and checkpointed per-model version history without fetching bytes;
- all public Kaggle model variations and their historical versions, retaining variation- and
  version-scoped metadata and download references without fetching model files;
- TorchGeo source-declared pretrained weight definitions and historical AllenNLP model archives, plus five first-party Fairseq NLP model zoos: RoBERTa, BART, XLM-R, mBART, and Transformer language models;
- first-party checkpoint registries for Scenic, StarDist, Coqui TTS, DGL-LifeSci (including generative models), Uni-MOF, OpenFold/OpenFold3, and Argus robotics, retaining exact source-native handles and checkpoint links;
- first-party PaddleNLP Taskflow UIE checkpoints, Pelican and Galaxea G05 VLA policies,
  FAIR Chemistry OCP legacy models, Dopamine checkpoint bundles, Kaldi model declarations,
  and the GPT4All model catalog;
- DeepChem Mol2Vec, Tencent GROVER, Microsoft VQ-Diffusion, AlphaChip, PaddleNLP
  sentiment, sherpa source-separation, and OpenVLA checkpoint inventories;
- ADMET-AI Chemprop ensembles, DIPY neuroimaging weights, PaddleNLP knowledge-mining
  checkpoints, Octo policies, and sherpa audio-tagging archives;
- public GitLab release-asset candidates from checkpointable project and release
  metadata traversal;
- the complete ACL Anthology XML manifest traversal, plus Oracle OCI pretrained-model and
  Azure Foundry models-sold-by-Azure catalogs; these are paper or hosted-offering evidence,
  not inferred weight claims;
- Google NeuralGCM and Pangu-Weather checkpoint declarations, CompVis Stable Diffusion first-stage and latent-diffusion README checkpoints, MACE-OFF23 and MACE foundation weights, and RFdiffusion pretrained checkpoints, alongside first-party Cellpose legacy checkpoint declarations, PyG GPSE checkpoint mappings, nnU-Net v1 task bundles, Diffusion Policy checkpoints, the Chemprop
  CheMeleon checkpoint, official MindSpore model-zoo checkpoint files, and RT-1 SavedModels;
- OpenAI RL Clarity's exact Procgen/IMPALA `.jd` pretrained-policy checkpoints, indexed from its public release page;
- Zenodo's public OAI-PMH DCAT stream for candidate neural-model records with matching checkpoint distributions;
- additional literal model/repository declarations from PyTorch Hub's curated listing, distinct
  from the separately enumerated TorchVision weight registry;
- exact ref-pinned `torch.hub.load` calls from first-party PyTorch Hub model cards, and NLTK model-data index bundles with source-declared IDs and archive links;
- additional first-party checkpoint indexes for AlphaFold's parameter archive, the GraphGPS
  GPS-deep release, Gensim's single-file pretrained models, OpenCV DNN sample-model URLs, and
  legacy MediaPipe `.tflite`/`.task` files; AWS Bedrock's separate Region matrix adds hosted
  availability and lifecycle evidence for exact model-card IDs;
- River's documented online anomaly-detection and clustering components as technique entries,
  separate from trained model/checkpoint claims;
- TensorFlow Model Garden's checkpoint-linked vision and NLP rows, T5X's declared
  pretrained checkpoints, and ProteinMPNN's official weight files, all pinned to
  first-party Git revisions and retained as references;
- SciSpaCy pretrained pipelines, PaddleX's source-declared model list, and WeNet's pretrained model registry;
- OpenML's public versioned flows as documented algorithm implementations,
  robomimic's declared policy checkpoints, and first-party LightGBM and
  imbalanced-learn technique pages;
- every xAI model returned by its Models API when `XAI_API_KEY` is configured,
  as account-visible serving evidence with the credential excluded from records;
- xAI image and video model-family availability APIs when `XAI_API_KEY` is configured,
  plus public Mistral lifecycle documentation;
- every configured-public-owner row returned by OpenAI's documented Models API, retaining
  its exact provider ID, owner, creation/shutdown metadata, and provider-resource link while
  excluding account-specific/private-owner rows before persistence;
- every public OpenAI model-documentation card structurally listed in the official all-models
  catalog, including documented deprecated models, using its URL path as an exact API-model ID;
  a later bounded card resolver retains declared snapshots as release evidence without querying
  an account or treating documentation navigation as a model-resource relation;
- every provider-available Claude row from Anthropic's paginated Models API, retaining exact
  provider ID, display name, release/capability metadata, and direct provider-resource link;
- every API-available Gemini resource returned by Google's documented paginated Models API,
  retaining its exact resource name, display/version/capability metadata, and declared
  base-model relation without embedding the required key in stored evidence or links;
- every active Groq-hosted model from its documented Models API, retaining its exact
  serving ID, owner and timestamp metadata as a credential-gated, current-availability
  observation rather than an ownership or weight-distribution claim;
- public lifecycle and retirement documentation for Gemini, OpenAI model snapshots, Groq, Azure Foundry, Anthropic, and AWS Bedrock, plus xAI API version/lifecycle metadata when `XAI_API_KEY` is configured;
- Together, Cerebras, DeepSeek, and SambaNova's credential-gated Models APIs, plus Cerebras'
  public model overview and Perplexity's public router-model table; API rows are current hosting
  availability, and public docs are provider offerings, not origin-checkpoint claims;
- every structurally linked public Mistral model-detail page, including its documented
  retired-model section, with the exact page worklist resolved in checkpointed batches
  for source-declared arXiv papers and Mistral Hugging Face cards only—not page navigation
  or an account-scoped API catalog;
- every exact Cohere API model name in its public structured overview tables, including
  documented live and retired/deprecated offerings, without querying the account API that
  can expose fine-tuned models;
- every first-party Amazon Bedrock model card structurally linked from its public catalog,
  retaining an AWS-scoped card identity and a source page that declares serving IDs,
  lifecycle information, and provider links, without enumerating custom account models;
- every entry in SageMaker JumpStart's public dated catalog, retaining its exact deployment
  model ID separately from the provider's display name across open-weight, proprietary, and
  built-in offerings, without calling account-scoped Hub or Marketplace discovery APIs;
- every first-party PyTorch Hub research-model card structurally listed in its public index,
  then resolved in checkpointed batches for its directly declared paper, exact source path,
  notebook/demo, Hugging Face Space, or checkpoint resource without guessing equivalence;
- every documented scikit-learn learning component in its current public API index and
  retained 0.18, 0.20, 0.22, 0.24, 1.0, 1.2, 1.4, 1.6, and 1.8 snapshots, across its
  estimator, neural-network, decomposition, clustering, transformation, and composition
  modules; the fully qualified API path is a shared source-native documented-technique
  identity across versions, never a claim of a trained release, paper, or weights;
- every structurally linked sktime forecasting, transformation, classification, regression,
  clustering, alignment, detection, parameter-estimation, and composition component in the
  first-party API, retaining the source-declared fully qualified class ID from each page
  anchor and the canonical documentation page without executing its package;
- every first-party aeon classification, regression, clustering, transformation,
  segmentation, and anomaly-detection component, retaining the exact qualified class path
  from its generated documentation page without importing the package or claiming a
  trained release;
- every Darts forecasting component structurally declared in its first-party index,
  including statistical, neural, foundation, wrapper, ensemble, and conformal classes,
  with its fully qualified class anchor and model-scoped documentation page retained;
- every PyOD detector structurally listed in its first-party cross-modality algorithm
  table, retaining its exact qualified implementation class and documentation page as
  documented anomaly-detection technique evidence, never as an asserted checkpoint;
- every PyGOD graph-anomaly detector structurally listed in its first-party generated
  API index, retaining the exact qualified detector class and documentation page as
  documented technique evidence without importing the package;
- every documented Statsmodels regression, generalized, discrete, multivariate,
  time-series, survival, robust, nonparametric, GAM, and miscellaneous model class in
  its current API plus retained 0.11--0.14 documentation snapshots, keyed by its exact
  qualified API path rather than presented as a trained release;
- every structurally linked tslearn clustering, early-classification, forecasting,
  matrix-profile, neighbor, neural-network, piecewise, shapelet, and SVM component in
  its first-party API, keyed by its exact qualified class path and retained as
  documented technique evidence rather than an implied checkpoint;
- every AutoGluon Tabular model class structurally declared in its first-party API,
  keyed by its exact qualified class anchor and retained as documented component
  evidence rather than a claimed trained release;
- every PyTorch Forecasting model class structurally linked in its first-party model
  index, keyed by its exact qualified API path and retained as documented forecasting
  component evidence rather than an implied checkpoint;
- every first-party GluonTS model-matrix row as a documented forecasting technique,
  retaining its source-declared paper and implementation links on the exact model entry
  without downloading any of those resources;
- every NeuralForecast architecture structurally linked in its first-party model index,
  retaining the source-native model-page slug and descriptive label as documented
  forecasting component evidence;
- every StatsForecast component structurally identified by its first-party generated
  API anchor, preserving exact classical-forecasting identities without maintaining a
  model-name list;
- every first-party PyTorch Tabular model configuration structurally listed in its
  model index, retaining the exact qualified configuration class as documented
  component evidence;
- every TorchGeo geospatial model family structurally linked in its first-party API,
  retaining the source-native model-page slug as documented component evidence;
- every Segmentation Models PyTorch architecture structurally declared in its
  first-party API, retaining its exact qualified architecture class as documented
  segmentation evidence;
- every public Zenodo record explicitly categorized as a `Model`, preserving its record and
  DOI identity, source-declared version/concept and related-resource links, and file references;
  its cross-domain category is evidence only, so neural-model candidates still require direct
  textual scope evidence;
- the complete versioned Papers With Code paper-to-code archive, restricted to its
  source-declared official implementation links and represented as concrete paper and
  code-repository artifacts; its separate methods dump is not trusted for model identity
  until independently validated (a separate, conservative arXiv-verified subset is
  enabled as documented identity evidence), while its arXiv-linked and paper-linked
  candidate tiers retain progressively lower-confidence method identities and its
  independent evaluation-table snapshot supplies paper-backed model-label candidates
  and reported-code links;
- the narrow, structured Wikidata class for machine-learning models, retained as
  documented identity evidence with a QID rather than as a release claim;
- a separate, broader Wikidata neural-network taxonomy tier for historical recall;
  its community classification is explicit evidence, never a provider release claim;
- the public Wikimedia neural-network-architecture category, held separately as
  documented historical evidence rather than as a release or provider assertion;
- structurally enumerated official Torchvision family pages and source-declared, pinned weight-enum
  releases; historical GluonCV classification, detection, segmentation, pose, action-recognition,
  and depth catalogs; Torchaudio pipeline pages and pinned source-declared checkpoint releases,
  Keras Applications, KerasHub family pages and pinned,
  source-declared KerasHub presets, Ultralytics model families,
  PyTorch Geometric, TorchDrug, DeepChem, Transformers, PaddleNLP transformer families with their
  directly declared paper links,
  PaddleNLP's published pretrained embedding identifiers, Sentence Transformers' curated
  pretrained repositories, spaCy's historical trained-pipeline packages, Stanza's versioned
  pretrained processor releases, and Diffusers, with the complete matched
  source document retained and no embedded model-name list;
- every structurally listed fastText Common Crawl/Wikipedia 300-dimensional vector release,
  retaining its exact language handle, paired binary/text artifact references, and the catalog's
  declared paper, source implementation, and license—without fetching vector bytes;
- every package row in Apple's public Core ML Model Gallery, retaining the exact package filename
  and its row-scoped, reference-only Core ML weights URL without equating a conversion package
  to an upstream technique or checkpoint;
- every source-declared pretrained release in TensorFlow's retained TF1/TF2 Detection, DeepLab,
  Slim, and Model Garden NLP tables, pinned to a repository commit with its exact artifact as a
  reference-only resource and with framework/task identities kept separate;
- every architecture imported by timm's maintained package registry at a pinned source
  revision, together with literal pretrained configuration, code, and exact declared
  Hugging Face or weight-resource links—without executing upstream code or downloading weights;
- every public PaddlePaddle ModelCenter family and its source-declared downloadable variants,
  pinned to one repository revision with direct implementation, documentation, paper, and
  artifact links where the catalog declares them;
- every literal ImageNet-series and PULC inference release exposed by PaddleClas's own
  command-line registry, pinned to its source revision with source-scoped code and direct,
  reference-only archive links—without executing PaddleClas or downloading an archive;
- every direct checkpoint row in PaddleDetection's versioned config model-zoo documents,
  parsed from one bounded, pinned source archive with model-card/configuration references
  and no project-code execution or checkpoint transfer;
- every direct checkpoint row in Paddle3D's first-party architecture model-zoo documents,
  preserving its source page, explicit configuration, and reference-only weight link under
  separate 3D-autonomy identities;
- every direct first-party model file declared in PaddleGAN's maintained English tutorials,
  retaining source-row papers or unambiguous single-tutorial paper/code references without
  transferring checkpoint bytes;
- every direct checkpoint row in PaddleSeg's independently versioned config model-zoo documents,
  with distinct source-native identities even where a segmentation architecture name overlaps;
- every direct checkpoint row in PaddleVideo's maintained English model-zoo pages, preserving
  its source page and explicit artifact/config references under a separate video-model namespace;
- every direct downloadable-model row in PaddleSlim's first-party compression, distillation, and
  NAS model-zoo table, pinned to its source revision with no artifact transfer;
- every source-declared PaddleRec recommendation-algorithm row, with its exact category, source
  implementation, documentation, online resource, and paper links kept in one documented entry;
- every structurally valid ESPnet Model Zoo record, pinned to its authoritative table revision,
  retaining speech task/corpus metadata and source-declared Zenodo or Hugging Face artifacts
  without transferring archive bytes;
- every direct-artifact row in Fairseq's retained wav2vec, wav2vec 2.0, and neural-machine-
  translation release tables, preserving each source-native model identifier and corpus context;
- named Seamless multilingual-model rows, retaining their source-declared Hugging Face model card,
  direct checkpoint, and evaluation evidence as separate model-scoped resources;
- the source-declared SONAR text-encoder and text-decoder checkpoint rows, while retaining tokenizer
  resources as non-model evidence rather than inventing a model identity;
- every exact MusicGen, MusicGen-Style, AudioGen, EnCodec, MAGNeT, JASCO, and AudioSeal model-card
  link in its designated first-party section, with an exact Hugging Face identity bridge and no
  claim that card URLs are binary weights;
- source-declared neural artifact rows from PaddleSpeech's released-model catalog and PaddleOCR's
  versioned V2 and current HTML-table model lists, each pinned to its source revision while excluding
  PaddleSpeech's explicitly non-neural N-gram section;
- every config/checkpoint pair in Detectron2's explicitly released, static Model Zoo mapping,
  pinned to its source revision with direct configuration, source, repository, and checkpoint
  references but no configuration or weight transfer;
- every direct artifact-bearing row in TensorFlow TPU, CenterNet, CenterTrack, AlphaPose, pycls, DINO,
  DINOv2, Barlow Twins, SwAV, I-JEPA, DiT, MAE, ESM, Segment Anything, SAM 2, ImageBind, VISSL, MaskFormer, Mask2Former, legacy Detectron, detrex, Detic, and X-AnyLabeling
  first-party historical Model Zoo tables, preserving experiment/table context and an exact source
  implementation plus artifact reference without fetching a model binary;
- BEiT, BEiT v2, and BEiT-3 source-native release declarations: the v1/v2 table rows retain
  their initialized-checkpoint context and unify the same documented release across task headings,
  while BEiT-3 admits only its explicit checkpoint-list models and excludes its tokenizer;
- DETR and ConvNeXt checkpoint rows, preserving each model's explicit backbone or resolution context
  and source-declared training-log evidence rather than merging repeated display names;
- first-party Swin Transformer V1/V2/SwinMLP release rows, preserving declared 1K/22K pretraining
  context, direct artifacts, and row-local training logs;
- independent DeiT, CaiT, ResMLP, and PatchConvNet historical release tables, with each model family
  kept in its own source namespace and direct checkpoint relation;
- PySlowFast and TimeSformer video-release rows, retaining first-party architecture/size or
  dataset context and direct checkpoint evidence;
- every literal direct-checkpoint handle in MMEngine's maintained OpenMMLab and MMClassification
  runtime registries, pinned to the source JSON revision with a model-scoped official implementation
  and artifact reference but no package import or checkpoint transfer;
- every literal direct-checkpoint handle in OpenAI's CLIP and Whisper source maps, AST-parsed at a
  pinned source revision without importing the package, invoking a downloader, or transferring a
  model binary;
- every source-declared model/config/weight row in OpenMMLab's current and archival project
  indexes—including final MMClassification 0.x, MMTracking, MMSelfSup, MMFewShot, MMFlow, and
  MMGeneration—each pinned to its own repository revision with exact code, paper, and checkpoint
  links where the project manifest declares them;
- every direct `.pth` checkpoint and Python configuration pair in MMHuman3D's first-party
  per-method model-zoo tables, kept separate from its broader supported-methods menu;
- every direct checkpoint in MMAction's retained original model-zoo document, preserving its
  historical action-recognition and detection plane separately from MMAction2;
- every explicit artifact-bearing model row in MMFashion's historical Model Zoo, with its
  first-party task/table context and reference-only hosted model links;
- every visible direct-artifact method row in OpenUnReID's historical person-re-identification
  Model Zoo, excluding commented documentation examples;
- every public exact-version BioImage.IO model resource, with verified model-card
  digests, citations, code/config links, weight formats, and release evidence;
- every public versioned MONAI Model Zoo bundle, pinned to the source repository revision
  and retaining its declared archive link and available checksum without downloading its bytes;
- every manifest in Intel's public OpenVINO Open Model Zoo at a pinned Git revision, retaining
  declared task/framework metadata, checksums, original model sources, and reference-only
  artifact URLs;
- OpenAlex publication and record-update windows, direct arXiv OAI-PMH, both public
  OpenReview note APIs, Crossref, Europe PMC, DataCite, the complete OpenAIRE Graph
  manifest, reusable full-text JATS from PMC OAI-PMH, and a range-read public arXiv
  historical-metadata snapshot through 2026-08-27;
- bioRxiv and medRxiv metadata plus their separate preprint-to-journal publication
  feeds, preserving every preprint version;
- the global OSF community-preprint API, spanning hosted rXiv communities without a
  provider or subject allowlist and preserving their individual OSF/DOI identities;
- the first-party EarthArXiv Janeway OAI-PMH stream, including current GeoAI and
  scientific-ML preprints;
- the complete Semantic Scholar dataset release/diff control plane, PubMed baseline
  and update-file manifests, every Common Crawl WET collection manifest, and every
  public GitHub event in each closed GH Archive hour selected by the durable
  activity watermark;
- direct first-party GitHub public-repository enumeration via its durable numeric-ID
  cursor, so inactive repositories do not need a recent event to enter the
  metadata/README frontier;
- every origin in the latest settled, officially listed Software Heritage full-graph
  ORC export, with exact GitHub origins projected to repository enrichment without a
  repository-name, topic, language, or model-term filter.

Semantic Scholar, PubMed, OpenAIRE, Common Crawl, GH Archive, and Software Heritage use
bulk control streams. `modelome bulk-load` transfers only a bounded number of their exact
declared shards, verifies the available digests and control identity, and writes
immutable sharded Parquet into a separate landing zone. Partial releases remain
resumable and are not advertised as complete. Common Crawl WET, GH Archive, and Software
Heritage loading stream bounded content without selecting domains, repositories, event
types, languages, or model names.

The repository also includes:

- a declarative, provider-neutral JSON catalog adapter with cursor or same-origin
  URL pagination, header credentials, model-card links, lineage, and opt-in release
  identity; no provider endpoint is enabled by default;
- generic introduction-cue extraction for paper/page text and scoped README-heading
  extraction for code/model-card artifacts, with no architecture vocabulary in runtime
  code, plus a bounded sealed Parquet candidate projection over every completed Semantic
  Scholar paper-state projection and every landed Common Crawl WET shard;
- a restart-safe Common Crawl search handoff that retains every crawled document in the
  immutable projection, while admitting only evidence-backed neural candidates and
  exact GitHub relations to the searchable store and repository frontier;
- a bounded URL frontier for links found in primary artifacts, including GitHub
  repository metadata/README enrichment, public web pages, and reference-only
  artifacts for directly linked weight files without downloading their contents;
  completed and failed fetches become eligible for a bounded daily refresh while
  policy-ignored targets remain terminal;
- an exact-identity GH Archive repository projection that turns every repository
  observed in public activity into frontier work, preserving the numeric GitHub
  repository ID, owner/name, event digest, hour, and locator before the generic
  README extractor decides whether the repository documents a neural model;
- a direct GitHub public-repository catalog source that pages its documented `since`
  cursor without a repository-name, owner, topic, language, or model-term filter,
  preserving its numeric identity and sending every observed public repository to the
  same metadata/README frontier;
- an immediate Software Heritage origin-shard projection that retains every archived
  origin in Parquet and structurally selects exact GitHub repository URLs for the same
  metadata/README frontier, including inactive repositories absent from recent activity;
- bounded alphaXiv full-text enrichment keyed only by arXiv IDs already discovered
  through primary sources; its search and recommendation tools are never discovery
  inputs;
- a deliberate `ingest-institutional-text` pathway for one paper an authorized user
  has already obtained under an applicable institutional license. It imports local
  UTF-8 extracted text as private evidence, never browser credentials or PDFs, and
  keeps that text out of metadata exports;
- a directory-backed Parquet store for immutable revisions, models, exact identifiers,
  aliases, releases, source checkpoints, relation claims, provenance, tombstones, run
  state, and dead letters;
- an opt-in entry workflow: `export-entry-seeds --link-current-resources` streams
  current admitted model claims, including direct URLs retained from the candidate
  record's text, and can attach exact URL/identifier evidence from other current records
  without building a global graph; `ingest-paper --input PAPER.json` composes one exact
  normalized paper observation into durable evidence, a bounded resource frontier, and a
  portable entry seed/plan without materializing entries. `entry-plan` reviews resource
  work without fetching, and `build-entry-corpus` writes a separate bundle only when
  explicitly invoked. The workflow keeps resources, releases, and loose tags together
  while using exact identifiers—not names—as cross-seed merge keys. `entry-readiness`
  reports whether current source completion and model claims support a future corpus build;
- a bounded, immutable artifact-relation projection that connects current papers, code,
  model cards, and weights through contextual URLs and exact artifact/model/release
  identities without joining on names;
- a non-destructive metadata-only export for public distribution: source-attributed
  Parquet identities and joins are retained, while raw source payloads, abstracts,
  full text, and card/README bodies are excluded;
- commands for search, detail, source status, aggregate statistics, historical
  backfill, complete arXiv/PMC bootstrap, bounded bulk loading, artifact linking, the
  composed daily run, acceptance-corpus coverage, and read-only benchmark evaluation.

Epoch AI is an external benchmark, not an ingestion source. The command
`modelome benchmark --name epoch` fetches its current public model corpus and measures
whether MODELOME discovered those models through independent sources. Epoch data is never
ingested or written to the Parquet store, so benchmark matches always require
independently discovered registry evidence. This pre-release codebase assumes a freshly
initialized store.

The system is not yet universal. Deeper package/weight registries, more provider and
code/weight repositories, additional domain corpora, lawful publisher text resolution,
and independent stratified extraction evaluation remain. Software Heritage greatly
widens the dormant-repository discovery plane, but it is a snapshot of its own holdings,
not proof that every repository ever created is present. Its origin table also has URLs,
not repository contents: current public GitHub metadata/README access is still needed for
model admission. Most importantly, the bulk landing zone is not yet a fully unified
searchable projection for all landed rows. Those capabilities stay dormant until the
entry-first pipeline has been validated; see [the source strategy](docs/sources.md) for
the detailed inventory and [Entry-first modelome](docs/entry-first.md) for operating
scope.

## Quick start

Python 3.12+ and [uv](https://docs.astral.sh/uv/) are recommended.

```bash
uv sync --extra dev
uv run modelome init
uv run modelome sources
uv run modelome benchmarks
uv run modelome lake-status
uv run modelome stats
```

The default store directory is `data/store`. Select another directory with
`--store PATH` or `MODELOME_STORE`; for example,
`uv run modelome --store /var/lib/modelome/store status`.

For a bounded source-adapter smoke test, which does not exhaust a corpus:

```bash
uv run modelome daily --max-pages 1 --max-new-shards 0 --bootstrap-max-pages 1 --no-frontier
```

Do not make `daily` a normal scheduled pass during the entry-first phase. It is a
future corpus-maintenance composition. `daily` advances every current feed, transfers a bounded number of bulk shards,
projects newly sealed GH Archive hours and each committed Software Heritage origin shard
into exact GitHub repository discoveries, publishes any complete Semantic Scholar
snapshot/diff projection, resumes the complete arXiv and PMC history bootstraps, enriches
exact arXiv IDs through alphaXiv when `ALPHAXIV_API_KEY` is set, enriches discovered
links, and publishes the current paper/code/card/weights relation projection. Each phase has
independent durable state. The searchable store and bulk landing zone both
contain Parquet only; neither requires or embeds a SQL database. The included systemd
timer invokes this one-shot command daily, then runs the read-only Epoch recall gate,
and catches up after host downtime; see [operations](docs/operations.md).

Set `GITHUB_TOKEN` to a least-privilege token when processing repository discoveries at
scale. MODELOME still accepts only public repositories by default; the token raises the REST
API quota and is never written to evidence.

The metadata export is suitable for a Hub dataset upload after reviewing its generated
README and source manifest. It refuses to overwrite a destination, preserves the
source commit and file hashes, and intentionally omits raw paper/card/source text.

For inclusive historical API windows:

```bash
uv run modelome backfill --source openalex --from 2012-01-01 --to 2012-12-31 --max-pages 100
uv run modelome backfill --source crossref --from 2012-01-01 --to 2012-12-31 --max-pages 100
uv run modelome backfill --source datacite --from 2012-01-01 --to 2012-12-31 --max-pages 100
uv run modelome backfill --source biorxiv --from 2012-01-01 --to 2012-12-31 --max-pages 100
uv run modelome backfill --source medrxiv-publications --from 2020-01-01 --to 2020-12-31 --max-pages 100
uv run modelome backfill --source osf-preprints --from 2018-01-01 --to 2018-12-31 --max-pages 100
uv run modelome backfill --source eartharxiv --from 2017-10-23 --to 2017-12-31 --max-pages 100
uv run modelome backfill --source hal --from 2002-09-23 --to 2002-12-31 --max-pages 100
```

Rerun the same command to resume. Each derived namespace is separate from its daily
source checkpoint, and a completed rerun is a no-op.

## Core model

```text
source catalogs / APIs
          | enumerate records or bounded windows
          v
 immutable artifact revisions ----> discovered URL frontier
          |                                  |
          | evidence-backed hints            +--> code/cards/public pages
          v
 models + exact identifiers + releases + relation claims
          |
          +--> searchable projection with supporting artifacts
          |
          +--> opt-in entry builder --> unified resource entries + loose tags
```

Papers, repositories, model cards, catalog rows, and provider pages are artifacts,
not aliases for one another. A release is a source-backed revision/checkpoint of a
model. Automatic model or release identity reuse requires an exact namespaced
identifier. Similar names are searchable but are never an automatic merge key.

The registry's current evidence tables remain conservative and source-oriented. The
offline [entry builder](docs/entries.md) is the opt-in path to its future public
surface: one unified technique/model entry with paper, code, checkpoint, card, data,
and evaluation resources plus loose tags. It merges only exact identifiers and never
requires downloading a corpus before processing one record.

## Measuring coverage

`modelome coverage` takes a read-only UTF-8 CSV with required `Model` and `Bucket`
columns. It exact-matches normalized canonical names or aliases that still have
active evidence from a current artifact revision. The manifest is never read by
discovery adapters, and its bucket labels do not become registry classifications.

```bash
uv run modelome coverage --manifest tests/fixtures/coverage_models.csv
```

The bundled fixture is an acceptance corpus spanning classic vision, GANs,
transformers, image diffusion, scientific diffusion, and non-diffusion neural models
in biology, medicine, chemistry, physics, and materials.
It is a sentinel set, not evidence that every member of those families has been
found. The command exits with status 1 when any expected model is missing and
also reports source exhaustion/counts, model identifier coverage, relation state,
and frontier state from the inspected store.

For the live Epoch benchmark:

```bash
uv run modelome benchmark --name epoch --minimum-recall 1.0
```

`modelome benchmarks` lists configured benchmarks. The Epoch command downloads the
current benchmark corpus, evaluates the existing registry without modifying it, and
exits nonzero when recall is below the requested threshold. The default `modelome sync`
does not fetch Epoch. Keeping benchmark data entirely outside discovery prevents the
answer key from contaminating the system it is intended to evaluate.

## Development

```bash
uv run pytest
uv run ruff check .
```

Source mechanics belong in `config/sources.toml` and `src/modelome/sources`. Expected
model/family names belong only in external validation inputs and tests. A new
adapter should include fixtures for pagination, checkpoint recovery, idempotency,
schema drift, malformed records, credential handling where applicable, and
evidence locators.
