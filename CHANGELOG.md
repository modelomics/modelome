# Changelog

## 0.1.0

First release of Modelome, an evidence-backed registry toolkit for research papers,
techniques, models, releases, and related resources.

### Included

- A Python 3.12+ package and `modelome` command for source discovery, search, inspection,
  status reporting, and metadata export.
- Resumable source adapters backed by an append-only Parquet evidence store, with source
  provenance, identifiers, and run status retained for review.
- Bounded paper-ingestion and resource-link workflows, plus explicit entry planning and
  entry-corpus build commands.
- Dataset bundle builds with manifests and checksums, and separate verified bulk-data
  landing and projection operations.
- A bundled, configurable source catalog spanning scholarly metadata, model hosts,
  framework catalogs, and first-party model/checkpoint registries.

### First-run limits

- A fresh run is not a complete census of models or papers. Each source has its own
  pagination, history, and credential limits; `--max-pages` bounds sync work and a
  budget-limited run can remain partial.
- Sources are not all queried automatically as an unbounded crawl. Paper ingestion starts
  from one supplied, exact paper observation, and link enrichment is bounded.
- Dataset exports contain admitted evidence available from the requested work. A
  `complete` build means that work finished; it does not establish coverage of all
  upstream records.
- Some sources require credentials or optional dependencies. Configure them as described
  in `.env.example`; credentials are not loaded from `.env` automatically.
